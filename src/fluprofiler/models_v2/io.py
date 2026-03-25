"""
Typed I/O contracts for v2 fluProfiler models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class BatchInput:
    """
    Unified model input for v2 models.

    Notes:
    - matrices keys are channel names (e.g. "serum_HA", "virus_HA")
    - masks are optional; model may infer fallback masks when absent
    - labels are normalized to shape (B,) by adapter or model boundary
    """

    matrices: Dict[str, torch.Tensor]
    matrix_masks: Optional[Dict[str, torch.Tensor]] = None
    passage_tokens: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelOutput:
    """
    Unified model output for v2 models.
    """

    logits: torch.Tensor
    pred: torch.Tensor
    loss: Optional[torch.Tensor] = None
    extras: Dict[str, Any] = field(default_factory=dict)


def normalize_labels_1d(labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Normalize labels to shape (B,) when labels are provided.
    Accepts legacy (B, 1) and flattens to 1D.
    """
    if labels is None:
        return None
    if labels.ndim == 1:
        return labels
    if labels.ndim == 2 and labels.shape[-1] == 1:
        return labels.view(-1)
    return labels.view(-1)
