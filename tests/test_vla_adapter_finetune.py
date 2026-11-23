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


# --- resuming from a saved checkpoint (--resume_from) ---

from flower.vla_adapter.finetune_dropout import (  # noqa: E402
    UPSTREAM_RENAMES,
    checkpoint_step,
    checkpoint_vlm,
    resume_argv,
)


def _checkpoint(tmp_path, step=50000):
    ckpt = tmp_path / f"configs+libero_10_no_noops+b8--image_aug--subset--{step}_chkpt"
    ckpt.mkdir()
    for name in ("action_head", "proprio_projector"):
        (ckpt / f"{name}--{step}_checkpoint.pt").write_bytes(b"")
    return ckpt


def test_checkpoint_step_comes_from_the_dir_name_and_needs_the_head_files(tmp_path):
    assert checkpoint_step(_checkpoint(tmp_path)) == 50000
    with pytest.raises(ValueError):
        checkpoint_step(tmp_path / "not-a-checkpoint")
    bare = tmp_path / "x--75000_chkpt"
    bare.mkdir()
    with pytest.raises(FileNotFoundError):
        checkpoint_step(bare)


def test_resume_argv_adds_upstream_resume_flags_and_shifts_the_decay(tmp_path):
    ckpt = _checkpoint(tmp_path)
    argv = ["--max_steps", "100005", "--num_steps_before_decay", "100000", "--batch_size", "8"]
    out = resume_argv(argv, ckpt)
    assert out[: out.index("--resume")] == [
        "--max_steps",
        "100005",
        "--num_steps_before_decay",
        "50000",
        "--batch_size",
        "8",
    ]
    assert out[out.index("--resume") :] == ["--resume", "True", "--resume_step", "50000", "--resum_vla_path", str(ckpt)]


def test_resume_argv_decay_already_passed_starts_decayed(tmp_path):
    out = resume_argv(["--num_steps_before_decay=40000"], _checkpoint(tmp_path))
    assert out[0] == "--num_steps_before_decay=0"


def test_resume_argv_without_a_decay_flag_is_refused(tmp_path):
    with pytest.raises(ValueError):
        resume_argv(["--max_steps", "100005"], _checkpoint(tmp_path))


def test_checkpoint_vlm_serves_the_merged_weights_unrenamed(tmp_path):
    from safetensors.torch import save_file

    tensors = {
        "vision_backbone.featurizer.blocks.0.ls1.scale_factor": torch.ones(2),
        "language_model.model.layers.0.mlp.up_proj.weight": torch.arange(4.0).reshape(2, 2),
        "projector.fc1.weight": torch.zeros(2, 2),
        "action_queries.weight": torch.full((3, 2), 7.0),
    }
    ckpt = tmp_path / "x--50000_chkpt"
    ckpt.mkdir()
    save_file(tensors, ckpt / "model.safetensors")
    state = checkpoint_vlm(ckpt)("ignored/vlm/path", hf_token="", load_for_training=True).state_dict()
    assert set(state) == set(tensors) and all(torch.equal(state[k], tensors[k]) for k in tensors)
    for key in state:  # upstream's rename_state_dict_keys (finetune.py) must leave every key as is
        assert not any(old in key for old, _ in UPSTREAM_RENAMES), key


def test_upstream_renames_match_finetune_py():
    import ast

    from flower.vla_adapter.finetune_dropout import VLA_ADAPTER

    source = (VLA_ADAPTER / "vla-scripts" / "finetune.py").read_text()
    start = source.index("replace_map = [")
    literal = source[start + len("replace_map = ") : source.index("]", start) + 1]
    assert tuple(ast.literal_eval(literal)) == UPSTREAM_RENAMES
