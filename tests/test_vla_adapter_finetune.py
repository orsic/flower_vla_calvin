"""flower/vla_adapter/finetune_dropout.py: the run_forward_pass wrapper around upstream finetune.py."""

from types import SimpleNamespace

import pytest
import torch

from flower.vla_adapter.finetune_dropout import dropout_forward_pass, pop_flag
from flower.vla_adapter.masked_forward import IGNORE_INDEX


def test_pop_flag_removes_both_spellings_and_keeps_the_rest():
    assert pop_flag(["--a", "1", "--modality_dropout", "none", "--b"], "modality_dropout", "subset") == (
        "none",
        ["--a", "1", "--b"],
    )
    assert pop_flag(["--modality_dropout=none", "--b"], "modality_dropout", "subset") == ("none", ["--b"])
    assert pop_flag(["--b"], "modality_dropout", "subset") == ("subset", ["--b"])


class _Model:
    def _build_multimodal_attention(self, *args):
        return "upstream"


def _batch(B=4):
    labels = torch.full((B, 6), IGNORE_INDEX)
    labels[:, 3:] = 7
    return {"labels": labels, "attention_mask": torch.ones(B, 6, dtype=torch.long)}


def _setup(training=True):
    model = _Model()
    vla = SimpleNamespace(training=training, module=model)  # stands in for DDP(model)
    head = SimpleNamespace(module=SimpleNamespace(predict_action="upstream"))
    seen = {}

    def original(vla_, action_head, proprio_projector, batch, *args, **kwargs):
        seen["masked"] = "_build_multimodal_attention" in vars(model)
        seen["head_patched"] = callable(action_head.module.predict_action)
        seen["kwargs"] = kwargs
        return "loss", {"loss_value": 1.0}

    return vla, head, original, seen, model


def test_subset_injects_masks_only_during_the_call(monkeypatch):
    import flower.vla_adapter.masked_forward as mf

    vla, head, original, seen, model = _setup()
    monkeypatch.setattr(mf, "unwrap", lambda v: v.module)
    out = dropout_forward_pass(original, "subset")(vla, head, None, _batch(), cfg="x")
    assert out == ("loss", {"loss_value": 1.0})
    assert seen == {"masked": True, "head_patched": True, "kwargs": {"cfg": "x"}}
    assert "_build_multimodal_attention" not in vars(model) and head.module.predict_action == "upstream"


@pytest.mark.parametrize("strategy,training", [("none", True), ("subset", False)])
def test_no_dropout_and_validation_run_upstream_as_is(strategy, training):
    vla, head, original, seen, _ = _setup(training)
    dropout_forward_pass(original, strategy)(vla, head, None, _batch())
    assert seen["masked"] is False and seen["head_patched"] is False


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        dropout_forward_pass(lambda *a: None, "dirichlet")
