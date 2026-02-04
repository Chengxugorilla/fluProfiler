"""
Loss functions and metrics for fluProfiler models.

Contains masked loss implementations and loss function creation utilities.
"""

import torch
import torch.nn as nn
import math


class _MaskedLoss(nn.Module):
    """Base class for masked losses"""

    def __init__(self, reduction='mean', ignore_nans=True, ignore_value=-100.0):
        super().__init__()
        self.reduction = reduction
        self.ignore_nans = ignore_nans
        self.ignore_value = ignore_value

    def forward(self, pred, target, mask=None):
        """Compute a loss between pred and target for given mask.
        Note that this implementation is faster than loss(pred[mask], target[mask])
        for a given loss, and is nan-proof."""
        if mask is None and self.ignore_value is not None:
            mask = target != self.ignore_value
        elif mask is None:
            mask = torch.ones_like(target, dtype=bool)
        target_proxy = target
        if self.ignore_nans:
            target_proxy = target.clone()
            nans = torch.isnan(target)
            if nans.any():
                with torch.no_grad():
                    mask = mask & ~nans
                    target_proxy[nans] = 0
        # full_loss = self.criterion(pred, target_proxy)
        if self.reduction == 'meanmean' and pred.ndim == 3 and pred.shape[-1] == 1:
            # token-level binary classification
            # pred: n , seq_len, 1 -> n * seq_len
            # target: n, seq_len -> n * seq_len
            full_loss = self.criterion(pred.view(-1), target_proxy.view(-1))
            full_loss = torch.reshape(full_loss, (-1, pred.shape[1]))
        elif self.reduction == 'meanmean' and pred.ndim == 3:
            if target.ndim == 3:
                # token-level regression
                # pred: n , seq_len, label_size -> n * seq_len * label_size
                # target: n, seq_len, label_size -> n * seq_len * label_size
                full_loss = self.criterion(pred.view(-1), target_proxy.view(-1))
                full_loss = torch.reshape(full_loss, (-1, pred.shape[1], pred.shape[-1]))
            else:
                # token-level multi classification
                # pred: n , seq_len, label_size -> n * seq_len, label_size
                # target: n, seq_len -> n * seq_len
                full_loss = self.criterion(pred.view(-1, pred.shape[-1]), target_proxy.view(-1))
                full_loss = torch.reshape(full_loss, (-1, pred.shape[1]))
        elif self.reduction == 'meanmean' and pred.ndim == 2 and target.ndim == 2:
            # seq-level multi label
            # pred: n , label_size -> n * label_size
            # target: n, label_size -> n * label_size
            full_loss = self.criterion(pred.view(-1), target_proxy.view(-1))
            full_loss = torch.reshape(full_loss, (-1, pred.shape[1]))
        elif self.reduction == 'meanmean':
            self.reduction = "mean"
            full_loss = self.criterion(pred, target_proxy)
        else:
            full_loss = self.criterion(pred, target_proxy)

        full_loss[~mask] = 0

        if self.reduction == 'none':
            return full_loss
        if self.reduction == 'sum':
            return full_loss.sum()
        if self.reduction == 'mean':
            return full_loss.sum() / (mask.to(full_loss.dtype).sum() + 1e-12)
        if self.reduction == 'meanmean':
            if mask.ndim == 3:
                mask_sum = mask.to(full_loss.dtype).sum(dim=-1)
                full_loss = full_loss.sum(dim=-1) / (mask_sum + 1e-12)
                mask_sum = mask_sum.to(torch.bool).sum(dim=-1)
                full_loss = full_loss.sum(dim=-1) / (mask_sum + 1e-12)
                mask_sum = mask_sum.to(torch.bool).sum()
                loss = full_loss.sum() / (mask_sum + 1e-12)
            else:
                mask_sum = mask.to(full_loss.dtype).sum(dim=-1)
                loss = torch.sum(full_loss.sum(dim=-1) / (mask_sum + 1e-12)) / (mask_sum.to(torch.bool).sum() + 1e-12)
            return loss
        if self.reduction in ["summean", "meansum"]:
            if mask.ndim == 3:
                mask_sum = mask.to(full_loss.dtype).sum(dim=-1)
                full_loss = full_loss.sum(dim=-1)
                mask_sum = mask_sum.to(torch.bool).sum(dim=-1)
                full_loss = full_loss.sum(dim=-1) / (mask_sum + 1e-12)
                mask_sum = mask_sum.to(torch.bool).sum()
                loss = full_loss.sum() / (mask_sum + 1e-12)
            else:
                mask_sum = mask.to(full_loss.dtype).sum(dim=-1)
                loss = full_loss.sum() / (mask_sum.to(torch.bool).sum() + 1e-12)
            return loss
        return full_loss


class MaskedMSELoss(_MaskedLoss):
    """Masked MSE loss"""
    def __init__(self, reduction='mean', ignore_nans=True, ignore_value=-100.0):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans, ignore_value=ignore_value)
        self.criterion = nn.MSELoss(reduction='none')


def create_loss_function(config,
                         args,
                         hidden_size,
                         classifier_size,
                         sigmoid,
                         output_mode,
                         num_labels,
                         loss_type,
                         ignore_index=-100,
                         return_types=["dropout", "hidden_layer", "hidden_act", "classifier", "output", "loss"]
                         ):
    '''
    create the output layer and loss layer
    :param hidden_size:
    :param config:
    :param args:
    :param classifier_size:
    :param sigmoid:
    :param output_mode:
    :param num_labels:
    :param loss_type:
    :param ignore_index:
    :param return_types:
    :return:
    '''
    dropout, hidden_layer, hidden_act, classifier, output, loss_fct = None, None, None, None, None, None
    if "dropout" in return_types:
        if hasattr(config, "classifier_dropout_prob"):
            dropout = nn.Dropout(config.classifier_dropout_prob)
        elif hasattr(config, "classifier_dropout"):
            dropout = nn.Dropout(config.classifier_dropout)
        elif hasattr(config, "dropout_prob"):
            dropout = nn.Dropout(config.dropout_prob)
        else:
            dropout = nn.Dropout(0.1)

    if "hidden_layer" in return_types:
        hidden_layer = nn.Linear(hidden_size, classifier_size, bias=True)
        hidden_size = classifier_size

    if "hidden_act" in return_types:
        if hasattr(args, "classifier_activate_func"):
            hidden_act = create_activate(args.classifier_activate_func)
        elif hasattr(config, "classifier_activate_func"):
            hidden_act = create_activate(config.classifier_activate_func)
        elif hasattr(args, "hidden_act"):
            hidden_act = create_activate(args.hidden_act)
        elif hasattr(config, "hidden_act"):
            hidden_act = create_activate(config.hidden_act)

    if "classifier" in return_types:
        if sigmoid:
            if output_mode in ["binary_class", "binary-class"]:
                classifier = nn.Linear(hidden_size, 1, bias=True)
            else:
                classifier = nn.Linear(hidden_size, num_labels, bias=True)
        else:
            classifier = nn.Linear(hidden_size, num_labels, bias=True)

    if "output" in return_types:
        if sigmoid or output_mode in ["multi_label", "multi-label", "binary_class", "binary-class"]:
            output = nn.Sigmoid()
        elif output_mode in ["multi_class", "multi-class"]:
            output = nn.Softmax(dim=-1)
        else:
            output = None

    if "loss" in return_types:
        # positive weight
        if hasattr(args, "pos_weight") and args.pos_weight:
            pos_weight = args.pos_weight
        elif hasattr(config, "pos_weight") and config.pos_weight:
            pos_weight = config.pos_weight
        else:
            pos_weight = None

        if hasattr(args, "weight") and args.weight is not None:
            weight = args.weight
        elif hasattr(config, "weight") and config.weight is not None:
            weight = config.weight
        else:
            weight = None

        reduction = config.loss_reduction if hasattr(config, "loss_reduction") else "meanmean"

        loss_fct = MaskedMSELoss(reduction=reduction, ignore_nans=True,
                                    ignore_value=ignore_index * 1.0 if ignore_index else None)
    return dropout, hidden_layer, hidden_act, classifier, output, loss_fct


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


def create_activate(activate_func):
    '''
    create activate function by activate func name
    :param activate_func:
    :return:
    '''
    if activate_func:
        activate_func = activate_func.lower()
    if activate_func == "tanh":
        return nn.Tanh()
    elif activate_func == "relu":
        return nn.ReLU()
    elif activate_func == "leakyrelu":
        return nn.LeakyReLU()
    elif activate_func == "gelu":
        return nn.GELU()
    elif activate_func == "gelu_new":
        return NewGELUActivation()
    else:
        return nn.Tanh()