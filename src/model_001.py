import logging
import torch
import copy
import os
import math
# sys.path.append("..")
# sys.path.append("../..")
# sys.path.append("../../..")
# sys.path.append("../../../src")


# from src.common.pooling import create_pooler
# from src.common.loss import create_loss_function, create_activate
# from src.common.loss import create_activate
# from src.utils import *
# from src.common.multi_label_metrics import *
# from src.common.metrics import *

logger = logging.getLogger(__name__)

from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.configuration_bert import BertConfig
from transformers.configuration_utils import PretrainedConfig
import torch.nn as nn


class fluProfiler_Config(PretrainedConfig):
    def __init__(self, pad_token_id: int = 0, **kwargs):
        super().__init__(pad_token_id=pad_token_id, **kwargs)


def create_pooler(pooler_type, config, args):
    '''
    pooler building
    :param config:
    :param args:
    :return:
    '''
    if pooler_type == "seq":
        pooling_type = args.seq_pooling_type
        hidden_size = config.hidden_size
    else:
        pooling_type = args.matrix_pooling_type
        hidden_size = config.embedding_input_size

    return GlobalMaskValueAttentionPooling1D(embed_size=hidden_size)

class GlobalMaskValueAttentionPooling1D(nn.Module):
    def __init__(self, embed_size, units=None, use_additive_bias=False, use_attention_bias=False):
        super(GlobalMaskValueAttentionPooling1D, self).__init__()
        self.embed_size = embed_size
        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        self.units = units if units else embed_size 
        self.U = nn.Parameter(torch.Tensor(self.embed_size, self.units))
        self.V = nn.Parameter(torch.Tensor(self.embed_size, self.units))
        self.W = nn.Parameter(torch.Tensor(self.units, self.embed_size))

        nn.init.trunc_normal_(self.U, std=0.01)
        nn.init.trunc_normal_(self.V, std=0.01)
        nn.init.trunc_normal_(self.W, std=0.01)

        if self.use_additive_bias:
            self.b1 = nn.Parameter(torch.Tensor(self.units))
            nn.init.trunc_normal_(self.b1, std=0.01)
        if self.use_attention_bias:
            self.b2 = nn.Parameter(torch.Tensor(self.embed_size))
            nn.init.trunc_normal_(self.b2, std=0.01)

    def forward(self, x, mask=None, save_attention_path=None):
        # (B, Len, Embed) x (Embed, Units) = (B, Len, Units)
        q = torch.matmul(x, self.U)
        k = torch.matmul(x, self.V)

        if self.use_additive_bias:
            h = torch.tanh(q + k + self.b1)
        else:
            h = torch.tanh(q + k)

        # (B, Len, Units) x (Units, Embed) = (B, Len, Embed)
        if self.use_attention_bias:
            e = torch.matmul(h, self.W) + self.b2
        else:
            e = torch.matmul(h, self.W)
        
        if mask is not None:
            attention_probs = nn.Softmax(dim=1)(e + torch.unsqueeze((1.0 - mask) * -10000, dim=-1))
        else:
            attention_probs = nn.Softmax(dim=1)(e)

        if save_attention_path is not None:
            # save_attention_path = '/data/chenyihao/attention_prob/'
            filenames = os.listdir(save_attention_path)
            if len(filenames) == 0:
                max_num = 0
            else:
                max_num = max([int(os.path.splitext(file)[0]) for file in filenames])
            torch.save(attention_probs, save_attention_path + f'{max_num + 1}.pth')

        x = torch.sum(attention_probs * x, dim=1)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.embed_size) + ' -> ' + str(self.embed_size) + ')'

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
        """
        if not (target.size() == pred.size()):
            warnings.warn(
                "Using a target size ({}) that is different to the pred size ({}). "
                "This will likely lead to incorrect results due to broadcasting. "
                "Please ensure they have the same size.".format(
                    target.size(), pred.size()),
                stacklevel=2,
            )
        """
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
        # print("mask shape")
        # print(mask.shape)
        if self.reduction == 'meanmean' and pred.ndim == 3 and pred.shape[-1] == 1:
            # token-level binary classification
            # pred: n , seq_len, 1 -> n * seq_len
            # target: n, seq_len -> n * seq_len
            full_loss = self.criterion(pred.view(-1), target_proxy.view(-1))
            full_loss = torch.reshape(full_loss, (-1, pred.shape[1]))
            # print("ok1")
        elif self.reduction == 'meanmean' and pred.ndim == 3:
            if target.ndim == 3:
                # token-level regression
                # pred: n , seq_len, label_size -> n * seq_len * label_size
                # target: n, seq_len, label_size -> n * seq_len * label_size
                full_loss = self.criterion(pred.view(-1), target_proxy.view(-1))
                full_loss = torch.reshape(full_loss, (-1, pred.shape[1], pred.shape[-1]))
                # print("ok21")
            else:
                # token-level multi classification
                # pred: n , seq_len, label_size -> n * seq_len, label_size
                # target: n, seq_len -> n * seq_len
                full_loss = self.criterion(pred.view(-1, pred.shape[-1]), target_proxy.view(-1))
                full_loss = torch.reshape(full_loss, (-1, pred.shape[1]))
                # print("ok22")
        elif self.reduction == 'meanmean' and pred.ndim == 2 and target.ndim == 2:
            # seq-level multi label
            # pred: n , label_size -> n * label_size
            # target: n, label_size -> n * label_size
            full_loss = self.criterion(pred.view(-1), target_proxy.view(-1))
            full_loss = torch.reshape(full_loss, (-1, pred.shape[1]))
            # print("ok3")
        elif self.reduction == 'meanmean':
            self.reduction = "mean"
            full_loss = self.criterion(pred, target_proxy)
            # print("ok4")
        else:
            full_loss = self.criterion(pred, target_proxy)
            # print("ok5")

        full_loss[~mask] = 0
        """
        if not mask.any():
            warnings.warn("Evaluation mask is False everywhere, this might lead to incorrect results.")
            print(full_loss.sum(), mask.to(full_loss.dtype).sum())
        """
        if self.reduction == 'none':
            return full_loss
        if self.reduction == 'sum':
            return full_loss.sum()
        if self.reduction == 'mean':
            """
            print("mask:")
            print(mask.to(full_loss.dtype).sum(dim=-1))
            print(mask.to(full_loss.dtype).sum())
            """
            return full_loss.sum() / (mask.to(full_loss.dtype).sum() + 1e-12)
        if self.reduction == 'meanmean':
            if mask.ndim == 3:
                mask_sum = mask.to(full_loss.dtype).sum(dim=-1)
                """
                print("mask:")
                print(mask_sum)
                """
                full_loss = full_loss.sum(dim=-1) / (mask_sum + 1e-12)
                mask_sum = mask_sum.to(torch.bool).sum(dim=-1)
                # print(mask_sum)
                full_loss = full_loss.sum(dim=-1) / (mask_sum + 1e-12)
                mask_sum = mask_sum.to(torch.bool).sum()
                # print(mask_sum)
                loss = full_loss.sum() / (mask_sum + 1e-12)
            else:
                mask_sum = mask.to(full_loss.dtype).sum(dim=-1)
                """
                print("mask:")
                print(mask_sum)
                print(mask_sum.to(torch.bool).sum())
                """
                loss = torch.sum(full_loss.sum(dim=-1) / (mask_sum + 1e-12)) / (mask_sum.to(torch.bool).sum() + 1e-12)
            # print(full_loss.sum() / (mask.to(full_loss.dtype).sum() + 1e-12), loss)
            return loss
        if self.reduction in ["summean", "meansum"]:
            if mask.ndim == 3:
                mask_sum = mask.to(full_loss.dtype).sum(dim=-1)
                """
                print("mask:")
                print(mask_sum)
                """
                full_loss = full_loss.sum(dim=-1)
                mask_sum = mask_sum.to(torch.bool).sum(dim=-1)
                # print(mask_sum)
                full_loss = full_loss.sum(dim=-1) / (mask_sum + 1e-12)
                mask_sum = mask_sum.to(torch.bool).sum()
                # print(mask_sum)
                loss = full_loss.sum() / (mask_sum + 1e-12)
            else:
                mask_sum = mask.to(full_loss.dtype).sum(dim=-1)
                """
                print("mask:")
                print(mask_sum)
                print(mask_sum.to(torch.bool).sum())
                """
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

class BertPreTrainedModel(PreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class LucaQuadruple_final_dropout(BertPreTrainedModel):
    def __init__(self, config, args):
        super(LucaQuadruple_final_dropout, self).__init__(config)
        if config.seq_max_length is None and config.seq_max_length_a == config.seq_max_length_b == config.seq_max_length_c == config.seq_max_length_d:
            config.seq_max_length = config.seq_max_length_a
        if config.matrix_max_length is None and config.matrix_max_length_a == config.matrix_max_length_b == config.matrix_max_length_c == config.matrix_max_length_d:
            config.matrix_max_length = config.matrix_max_length_a
        if config.embedding_input_size is None and config.embedding_input_size_a == config.embedding_input_size_b == config.embedding_input_size_c == config.embedding_input_size_d:
            config.embedding_input_size = config.embedding_input_size_a

        self.num_labels = config.num_labels
        self.fusion_type = args.fusion_type if hasattr(args, "fusion_type") and args.fusion_type else "concat"
        self.output_mode = args.output_mode
        self.task_level_type = args.task_level_type
        self.prepend_bos = args.prepend_bos
        self.append_eos = args.append_eos

        self.seq_encoder, self.seq_pooler, \
        self.matrix_encoder, self.matrix_pooler = None, None, None, None
        self.encoder_type_list = [False, False, False]
        self.input_size_list = [0, 0, 0]
        self.linear_idx = [-1, -1, -1]

        self.matrix_dropout = nn.Dropout(p=0.1)
 

        self.input_size_list[1] = config.hidden_size

        new_config = copy.deepcopy(config)
        new_config.embedding_input_size = config.hidden_size
        self.matrix_pooler = create_pooler(pooler_type="matrix", config=new_config, args=args)
        
        
        self.encoder_type_list[1] = True
        self.linear_idx[1] = 0


        fc_size_list = [config.seq_fc_size, config.matrix_fc_size, config.vector_fc_size]
        all_linear_list = [None, None, None]
        self.output_size = [0, 0, 0]
        print("self.encoder_type_list:", self.encoder_type_list)
        for encoder_idx, encoder_flag in enumerate(self.encoder_type_list):
            if not encoder_flag:
                continue
            fc_size = fc_size_list[encoder_idx]
            input_size = self.input_size_list[encoder_idx]
            print("encoder_idx", encoder_idx, "input_size:", input_size)
            if fc_size is not None and len(fc_size) > 0:
                if isinstance(fc_size, list):
                    fc_size = [int(v) for v in fc_size]
                else:
                    fc_size = [int(fc_size)]
                linear_list = []
                for idx in range(len(fc_size)):
                    linear = nn.Linear(input_size, fc_size[idx])
                    linear_list.append(linear)
                    linear_list.append(create_activate(config.fc_activate_func))
                    input_size = fc_size[idx]
                all_linear_list[encoder_idx] = nn.ModuleList(linear_list)
                self.output_size[encoder_idx] = fc_size[-1]
            else:
                # 没有全连接层
                self.linear_idx[encoder_idx] = -1
                self.output_size[encoder_idx] = input_size
        all_linear_list = [linear for linear in all_linear_list if linear is not None]
        if all_linear_list is not None and len(all_linear_list) > 0:
            self.linear = nn.ModuleList(all_linear_list)

        last_hidden_size = sum(self.output_size)
        
        last_hidden_size = last_hidden_size
        self.dropout, self.hidden_layer, self.hidden_act, self.classifier, self.output, self.loss_fct = \
            create_loss_function(
                config,
                args,
                hidden_size=last_hidden_size * 4 + 256,
                classifier_size=args.classifier_size,
                sigmoid=args.sigmoid,
                output_mode=args.output_mode,
                num_labels=self.num_labels,
                loss_type=args.loss_type,
                ignore_index=args.ignore_index,
                return_types=["dropout", "hidden_layer", "hidden_act", "classifier", "output", "loss"]
            )
        
        self.Passage_encoder = nn.Sequential(
            nn.Embedding(num_embeddings=5, embedding_dim=256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

    def __forward__(
            self,
            input_ids,
            seq_attention_masks,
            token_type_ids,
            position_ids,
            vectors,
            matrices,
            matrix_attention_masks,
            save_attention_path
    ):  
        if matrices is not None:
            matrices = self.matrix_dropout(matrices)
            matrix_vector = self.matrix_pooler(matrices, mask=matrix_attention_masks, save_attention_path=save_attention_path)

            matrix_linear_idx = self.linear_idx[1]
            if matrix_linear_idx != -1:
                for i, layer_module in enumerate(self.linear[matrix_linear_idx]):
                    matrix_vector = layer_module(matrix_vector)

        concat_vector = matrix_vector

        return concat_vector

    def forward(
            self,
            input_ids_a=None, input_ids_b=None, input_ids_c=None, input_ids_d=None,
            position_ids_a=None, position_ids_b=None, position_ids_c=None, position_ids_d=None,
            token_type_ids_a=None, token_type_ids_b=None, token_type_ids_c=None, token_type_ids_d=None,
            seq_attention_masks_a=None, seq_attention_masks_b=None, seq_attention_masks_c=None, seq_attention_masks_d=None,
            vectors_a=None, vectors_b=None, vectors_c=None, vectors_d=None,
            matrices_a=None, matrices_b=None, matrices_c=None, matrices_d=None,
            matrix_attention_masks_a=None, matrix_attention_masks_b=None, matrix_attention_masks_c=None, matrix_attention_masks_d=None,
            strainPassCats=None, labels=None,
            save_concat_vector=None,
            save_attention_path=None,
            **kwargs
    ):  
        representation_vector_a = self.__forward__(
            input_ids_a,
            seq_attention_masks_a,
            token_type_ids_a,
            position_ids_a,
            vectors_a,
            matrices_a,
            matrix_attention_masks_a,
            save_attention_path
        )

        representation_vector_b = self.__forward__(
            input_ids_b,
            seq_attention_masks_b,
            token_type_ids_b,
            position_ids_b,
            vectors_b,
            matrices_b,
            matrix_attention_masks_b,
            save_attention_path
        )

        representation_vector_c = self.__forward__(
            input_ids_c,
            seq_attention_masks_c,
            token_type_ids_c,
            position_ids_c,
            vectors_c,
            matrices_c,
            matrix_attention_masks_c,
            save_attention_path
        )

        representation_vector_d = self.__forward__(
            input_ids_d,
            seq_attention_masks_d,
            token_type_ids_d,
            position_ids_d,
            vectors_d,
            matrices_d,
            matrix_attention_masks_d,
            save_attention_path
        )

        strainPassCats_vector = torch.mean(self.Passage_encoder(strainPassCats), dim=1)
        
        concat_vector = torch.concat([representation_vector_a, representation_vector_b, representation_vector_c, representation_vector_d], dim=1)
        
        if save_concat_vector is not None:
            filenames = os.listdir(save_concat_vector)
            if len(filenames):
                max_num = max([int(filename.split('.pth')[0]) for filename in filenames])
                torch.save(concat_vector, save_concat_vector + "/{}.pth".format(max_num+1))
            else:
                torch.save(concat_vector, save_concat_vector + "/1.pth")

        if self.dropout is not None:
            concat_vector = self.dropout(concat_vector)

        concat_vector = torch.concat([concat_vector, strainPassCats_vector], dim=1)

        concat_vector = self.hidden_layer(concat_vector)
        concat_vector = self.hidden_act(concat_vector)

        logits = self.classifier(concat_vector)
        output = logits

        outputs = [logits, output]
        if labels is not None:
            if self.output_mode in ["regression"]:
                if self.task_level_type not in ["seq_level"] and self.loss_reduction == "meanmean":
                    # logits: N, seq_len, 1
                    # labels: N, seq_len
                    loss = self.loss_fct(logits, labels)
                else:
                    # logits: N * seq_len
                    # labels: N * seq_len
                    loss = self.loss_fct(logits.view(-1), labels.view(-1))
            elif self.num_labels <= 2 or self.output_mode in ["binary_class", "binary-class"]:
                if self.task_level_type not in ["seq_level"] and self.loss_reduction == "meanmean":
                    # logits: N ,seq_len, 1
                    # labels: N, seq_len
                    loss = self.loss_fct(logits, labels.float())
                else:
                    # logits: N * seq_len * 1
                    # labels: N * seq_len
                    loss = self.loss_fct(logits.view(-1), labels.view(-1).float())

            outputs = [loss, *outputs]
        return outputs