"""
Model configuration classes for fluProfiler.
"""

from transformers.configuration_utils import PretrainedConfig


class fluProfiler_Config(PretrainedConfig):
    def __init__(self, pad_token_id: int = 0, **kwargs):
        super().__init__(pad_token_id=pad_token_id, **kwargs)