"""
Explicit HA-only v2 model.

This class is a thin wrapper around fluProfiler_HA_v2 with fixed HA channels.
"""

from __future__ import annotations

from .ha import fluProfiler_HA_v2


class fluProfiler_HA_only_v2(fluProfiler_HA_v2):
    """
    HA-only model with fixed channels:
    - serum_HA
    - virus_HA
    """

    def __init__(self, config, args):
        super().__init__(
            config=config,
            args=args,
            channel_names=("serum_HA", "virus_HA"),
            passage_vocab_size=6,
            passage_dim=256,
        )
