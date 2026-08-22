"""
Archetti-Horsfall reciprocal antigenic distance utilities.
"""

from __future__ import annotations

import math
from typing import Any


def _parse_uncensored_titer(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.startswith("<") or stripped.startswith(">"):
            return None
        value = stripped
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def compute_archetti_horsfall_distance(HAA, HBA, HBB, HAB) -> float | None:
    """
    Compute 0.5 * (log2(HAA/HBA) + log2(HBB/HAB)).

    Returns None when any titer is missing, non-positive, or censored.
    """

    parsed = [_parse_uncensored_titer(v) for v in (HAA, HBA, HBB, HAB)]
    if any(v is None for v in parsed):
        return None
    haa, hba, hbb, hab = parsed
    return 0.5 * (math.log2(haa / hba) + math.log2(hbb / hab))
