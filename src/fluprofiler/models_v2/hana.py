"""
v2 HANA model with unified I/O contract.
"""

from __future__ import annotations

from .ha import fluProfiler_HA_v2


class fluProfiler_HANA_v2(fluProfiler_HA_v2):
    """
    HANA model with fixed channels:
    - serum_HA
    - serum_NA
    - virus_HA
    - virus_NA
    """

    def __init__(self, config, args):
        super().__init__(
            config=config,
            args=args,
            channel_names=("serum_HA", "serum_NA", "virus_HA", "virus_NA"),
            passage_vocab_size=6,
            passage_dim=256,
        )
