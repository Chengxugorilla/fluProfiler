"""
New v2 model namespace (parallel to legacy models).
"""

from .io import BatchInput, ModelOutput, normalize_labels_1d
from .ha import fluProfiler_HA_v2
from .ha_only import fluProfiler_HA_only_v2
from .HA_only_distance import fluProfiler_HA_only_distance_v2
from .hana import fluProfiler_HANA_v2

__all__ = [
    "BatchInput",
    "ModelOutput",
    "normalize_labels_1d",
    "fluProfiler_HA_v2",
    "fluProfiler_HA_only_v2",
    "fluProfiler_HA_only_distance_v2",
    "fluProfiler_HANA_v2",
]
