"""
Helpers for constructing model argument namespaces from tracked config.
"""

from __future__ import annotations

from types import SimpleNamespace


DEFAULT_MODEL_ARGS = {
    "output_mode": "regression",
    "fusion_type": "concat",
    "task_level_type": "seq_level",
    "prepend_bos": False,
    "append_eos": False,
    "sigmoid": False,
    "loss_type": None,
    "ignore_index": -100,
    "weight_decay": 0.01,
    "beta1": 0.9,
    "beta2": 0.98,
    "adam_epsilon": 1e-6,
}


def _get_attr_or_key(obj, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def build_model_args(config=None, overrides=None):
    """
    Build a SimpleNamespace compatible with legacy fluProfiler model constructors.

    The repo historically loaded these values from ``configs/args.pkl``. This
    helper makes the same interface available from tracked JSON config instead.
    """

    values = dict(DEFAULT_MODEL_ARGS)
    values.update(
        {
            "classifier_size": _get_attr_or_key(config, "classifier_size", 256),
            "classifier_activate_func": _get_attr_or_key(
                config,
                "classifier_activate_func",
                _get_attr_or_key(config, "hidden_act", "gelu"),
            ),
            "hidden_act": _get_attr_or_key(config, "hidden_act", "gelu"),
            "pos_weight": _get_attr_or_key(config, "pos_weight", None),
            "weight": _get_attr_or_key(config, "weight", None),
        }
    )
    if overrides:
        values.update(overrides)
    return SimpleNamespace(**values)
