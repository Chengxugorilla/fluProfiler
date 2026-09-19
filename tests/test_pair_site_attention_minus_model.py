from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from fluprofiler.models.pair_site_attention_minus_model import (  # noqa: E402
    PairSiteAttentionMinusConfig,
    PairSiteAttentionMinusModel,
)
from fluprofiler.models.serum_gate_model import SerumGateBatch  # noqa: E402


def tiny_config(**overrides) -> PairSiteAttentionMinusConfig:
    values = {
        "hidden_size": 6,
        "max_site_length": 5,
        "site_proj_dim": 4,
        "d_model": 8,
        "num_layers": 2,
        "num_heads": 2,
        "transformer_ff_dim": 16,
        "site_attention_dim": 4,
        "dropout": 0.0,
        "passage_vocab_size": 3,
        "passage_pair_vocab_size": 9,
        "subtype_vocab_size": 2,
        "passage_dim": 2,
        "subtype_dim": 2,
        "predictor_hidden_dim": 8,
    }
    values.update(overrides)
    return PairSiteAttentionMinusConfig(**values)


def make_batch(
    *,
    reference_mask: torch.Tensor | None = None,
    query_mask: torch.Tensor | None = None,
    self_pair: bool = False,
) -> SerumGateBatch:
    torch.manual_seed(7)
    reference = torch.randn(1, 5, 6)
    if self_pair:
        query = reference[:, None].clone()
        query_count = 1
    else:
        query = torch.randn(1, 2, 5, 6)
        query_count = 2
    if reference_mask is None:
        reference_mask = torch.ones(1, 5)
    if query_mask is None:
        query_mask = torch.ones(1, query_count, 5)
    if query_mask.ndim == 2:
        query_mask = query_mask.unsqueeze(0)

    serum_passage = torch.tensor([1])
    query_passage = torch.ones(1, query_count, dtype=torch.long)
    passage_pair = serum_passage[:, None] * 3 + query_passage
    labels = torch.zeros(1, query_count) if self_pair else torch.tensor([[0.5, 1.5]])
    return SerumGateBatch(
        reference_ha=reference,
        query_ha=query,
        reference_ha_mask=reference_mask,
        query_ha_mask=query_mask,
        serum_passage=serum_passage,
        query_passage=query_passage,
        passage_pair=passage_pair,
        subtype=torch.tensor([1]),
        s_nagly=torch.zeros(1, query_count),
        labels=labels,
        query_mask=torch.ones(1, query_count),
    )


def test_pair_model_has_one_shared_site_projection():
    model = PairSiteAttentionMinusModel(tiny_config())

    projections = [
        name for name, _module in model.named_modules() if name.endswith("site_projection")
    ]

    assert projections == ["score_model.site_projection"]


def test_config_rejects_heads_that_do_not_divide_model_dimension():
    with pytest.raises(ValueError, match="d_model must be divisible by num_heads"):
        PairSiteAttentionMinusModel(tiny_config(d_model=7, num_heads=2))


def test_attention_uses_union_mask_and_normalizes_valid_sites():
    model = PairSiteAttentionMinusModel(tiny_config()).eval()
    batch = make_batch(
        reference_mask=torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]]),
        query_mask=torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0]]),
    )
    batch.query_ha = batch.query_ha[:, :1]
    batch.query_passage = batch.query_passage[:, :1]
    batch.passage_pair = batch.passage_pair[:, :1]
    batch.s_nagly = batch.s_nagly[:, :1]
    batch.labels = batch.labels[:, :1]
    batch.query_mask = batch.query_mask[:, :1]

    out = model(batch)
    alpha = out["query_site_attention"][0, 0]

    assert torch.allclose(alpha[3:], torch.zeros(2))
    assert torch.isclose(alpha[:3].sum(), torch.tensor(1.0))
    assert alpha[2] > 0
    assert torch.isfinite(out["nll_loss"])


def test_forward_returns_minus_scores_attention_and_finite_losses():
    model = PairSiteAttentionMinusModel(tiny_config()).eval()

    out = model(make_batch())

    assert out["mean"].shape == torch.Size([1, 2])
    assert out["log_var"].shape == torch.Size([1, 2])
    assert out["self_score"].shape == torch.Size([1, 2])
    assert out["query_score"].shape == torch.Size([1, 2])
    assert out["query_site_attention"].shape == torch.Size([1, 2, 5])
    assert out["self_site_attention"].shape == torch.Size([1, 2, 5])
    assert torch.isfinite(out["huber_loss"])
    assert torch.isfinite(out["nll_loss"])


def test_eval_self_pair_has_zero_minus_distance():
    model = PairSiteAttentionMinusModel(tiny_config()).eval()

    out = model(make_batch(self_pair=True))

    assert torch.allclose(out["mean"], torch.zeros_like(out["mean"]), atol=1e-6)
    assert torch.allclose(
        out["query_site_attention"],
        out["self_site_attention"],
        atol=1e-6,
    )


def test_distance_loss_backpropagates_through_all_pair_site_components():
    model = PairSiteAttentionMinusModel(tiny_config())

    out = model(make_batch())
    out["nll_loss"].backward()

    assert model.score_model.site_projection.weight.grad is not None
    assert model.score_model.transformer.layers[0].self_attn.in_proj_weight.grad is not None
    assert model.score_model.site_attention[0].weight.grad is not None
    assert model.score_model.prediction_head[0].weight.grad is not None
