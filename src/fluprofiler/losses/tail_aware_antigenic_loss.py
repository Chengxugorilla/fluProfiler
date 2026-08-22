"""
Tail-aware objective for HI-derived antigenic distance prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TailAwareLossConfig:
    smoothl1_beta: float = 0.5
    q_low: Optional[float] = None
    q_high: Optional[float] = None
    tail_quantile_low: float = 0.20
    tail_quantile_high: float = 0.80
    lambda_tail: float = 1.0
    lambda_dist: float = 0.1
    lambda_ah: float = 0.0
    lambda_reg: float = 0.01
    lambda_na_prior: float = 1.0
    rho_mean_weight: float = 0.1
    rho_var_weight: float = 0.01
    assay_l2_weight: float = 0.01
    na_prior_sigma: float = 0.05


class TailAwareAntigenicLoss(nn.Module):
    """
    SmoothL1 plus tail, distribution, and component regularization terms.
    """

    def __init__(
        self,
        smoothl1_beta: float = 0.5,
        q_low: float | None = None,
        q_high: float | None = None,
        tail_quantile_low: float = 0.20,
        tail_quantile_high: float = 0.80,
        lambda_tail: float = 1.0,
        lambda_dist: float = 0.1,
        lambda_ah: float = 0.0,
        lambda_reg: float = 0.01,
        lambda_na_prior: float = 1.0,
        rho_mean_weight: float = 0.1,
        rho_var_weight: float = 0.01,
        assay_l2_weight: float = 0.01,
        na_prior_sigma: float = 0.05,
    ):
        super().__init__()
        self.config = TailAwareLossConfig(
            smoothl1_beta=smoothl1_beta,
            q_low=q_low,
            q_high=q_high,
            tail_quantile_low=tail_quantile_low,
            tail_quantile_high=tail_quantile_high,
            lambda_tail=lambda_tail,
            lambda_dist=lambda_dist,
            lambda_ah=lambda_ah,
            lambda_reg=lambda_reg,
            lambda_na_prior=lambda_na_prior,
            rho_mean_weight=rho_mean_weight,
            rho_var_weight=rho_var_weight,
            assay_l2_weight=assay_l2_weight,
            na_prior_sigma=na_prior_sigma,
        )

    def _tail_thresholds(self, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.q_low is not None and self.config.q_high is not None:
            return (
                target.new_tensor(float(self.config.q_low)),
                target.new_tensor(float(self.config.q_high)),
            )
        return (
            torch.quantile(target.detach(), self.config.tail_quantile_low),
            torch.quantile(target.detach(), self.config.tail_quantile_high),
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        d_ha: torch.Tensor | None = None,
        d_ah: torch.Tensor | None = None,
        rho_ha: torch.Tensor | None = None,
        assay_l2: torch.Tensor | None = None,
        lambda_nagly: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pred = pred.view(-1).float()
        target = target.view(-1).float().to(pred.device)

        hi = F.smooth_l1_loss(pred, target, beta=self.config.smoothl1_beta)

        q_low, q_high = self._tail_thresholds(target)
        tail_mask = (target <= q_low) | (target >= q_high)
        if tail_mask.any():
            tail = (pred[tail_mask] - target[tail_mask]).pow(2).mean()
        else:
            tail = pred.new_tensor(0.0)

        dist = torch.abs(torch.std(pred, unbiased=False) - torch.std(target, unbiased=False))

        ah = pred.new_tensor(0.0)
        if d_ha is not None and d_ah is not None:
            d_ha = d_ha.view(-1).float().to(pred.device)
            d_ah = d_ah.view(-1).float().to(pred.device)
            ah = (d_ha - d_ah).pow(2).mean()

        reg = pred.new_tensor(0.0)
        if rho_ha is not None:
            log_rho = torch.log(rho_ha.view(-1).float().clamp_min(1e-8))
            reg = reg + self.config.rho_mean_weight * log_rho.mean().pow(2)
            reg = reg + self.config.rho_var_weight * torch.var(log_rho, unbiased=False)
        if assay_l2 is not None:
            reg = reg + self.config.assay_l2_weight * assay_l2.to(pred.device)

        na_prior = pred.new_tensor(0.0)
        if lambda_nagly is not None:
            sigma = max(float(self.config.na_prior_sigma), 1e-8)
            na_prior = (lambda_nagly.view(-1).float() / sigma).pow(2).mean()

        loss = (
            hi
            + self.config.lambda_tail * tail
            + self.config.lambda_dist * dist
            + self.config.lambda_ah * ah
            + self.config.lambda_reg * reg
            + self.config.lambda_na_prior * na_prior
        )

        return {
            "loss": loss,
            "hi": hi,
            "tail": tail,
            "dist": dist,
            "ah": ah,
            "reg": reg,
            "na_prior": na_prior,
        }
