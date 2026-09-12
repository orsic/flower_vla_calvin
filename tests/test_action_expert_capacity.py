"""
Tests for action-expert capacity knobs in flower/models/flower.py and
flower/models/networks/transformers.py.

Context: the pretrained checkpoint only covers 12 DiT blocks, but the shipped config uses
n_layers=18 -- the remaining 6 blocks are randomly initialized and appended after the
pretrained ones, trained at the VLM's learning rate. These tests exercise the knobs added
to make that capacity deliberate and trainable:
  1. dit_layer_sources -- maps each DiT module slot to a checkpoint layer index (or None),
     for every placement x init combination.
  2. zero_init_output_projections -- makes a fresh FlowBlock the identity function at init.
  3. FlowBlock's lora_dim/mlp_hidden_dim knobs, used to size extra blocks independently.
  4. FLOWERVLA._get_param_groups -- splits params into vlm / pretrained_expert /
     fresh_expert families, each with its own learning rate.
  5. TriStageLRScheduler -- scales each optimizer param group by its own base LR instead
     of overwriting every group with one scalar.
  6. dit_checkpoint_layout / FLOWERVLA._load_pretrained_weights -- pretrained_model_path is
     reused for two different checkpoint shapes (train time: the pretrained base, in
     checkpoint-index space; eval time: a fully-trained checkpoint already in this model's
     module-index space, per flower/evaluation/utils.py's load_mode_from_safetensor). These
     tests confirm each shape is recognized and, for the eval shape, that the extra blocks'
     real trained weights aren't zero-inited out from under them.

None of these touch Florence-2, so production code is exercised directly (no stand-ins).
"""

import pytest
import torch
from omegaconf import OmegaConf

from flower.models.utils import dit_layer_sources, dit_checkpoint_layout
from flower.models.networks.transformers import FlowBlock, zero_init_output_projections
from flower.models.flower import FLOWERVLA
from flower.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler


# ---------------------------------------------------------------------------
# 1. dit_layer_sources
# ---------------------------------------------------------------------------

def test_append_random_matches_todays_behavior():
    """append/random (and append/zero) is today's default: 12 pretrained blocks, then
    6 fresh (None) ones appended at the end."""
    assert dit_layer_sources(18, 12, "append", "random") == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] + [None] * 6


def test_append_zero_has_same_positions_as_append_random():
    assert dit_layer_sources(18, 12, "append", "zero") == dit_layer_sources(18, 12, "append", "random")


def test_append_copy_duplicates_the_last_half():
    assert dit_layer_sources(18, 12, "append", "copy") == list(range(12)) + [6, 7, 8, 9, 10, 11]


def test_interleave_random_spreads_extras_evenly():
    sources = dit_layer_sources(18, 12, "interleave", "random")
    assert sources == [0, 1, None, 2, 3, None, 4, 5, None, 6, 7, None, 8, 9, None, 10, 11, None]


def test_interleave_copy_duplicates_the_last_block_of_each_group():
    sources = dit_layer_sources(18, 12, "interleave", "copy")
    assert sources == [0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11]


@pytest.mark.parametrize("placement", ["append", "interleave"])
@pytest.mark.parametrize("init", ["random", "zero", "copy"])
def test_no_extras_returns_identity_mapping(placement, init):
    """n_layers == n_pretrained: no extra slots, regardless of placement/init."""
    assert dit_layer_sources(12, 12, placement, init) == list(range(12))


@pytest.mark.parametrize("placement", ["append", "interleave"])
def test_every_pretrained_index_appears_at_least_once(placement):
    """Every checkpoint layer must be used by some module slot (nothing silently dropped)."""
    sources = dit_layer_sources(18, 12, placement, "random")
    non_none = [s for s in sources if s is not None]
    assert set(non_none) == set(range(12))


def test_n_layers_below_pretrained_raises():
    with pytest.raises(ValueError):
        dit_layer_sources(6, 12, "append", "random")


def test_invalid_placement_and_init_raise():
    with pytest.raises(ValueError):
        dit_layer_sources(18, 12, "bogus", "random")
    with pytest.raises(ValueError):
        dit_layer_sources(18, 12, "append", "bogus")


# ---------------------------------------------------------------------------
# 1b. dit_checkpoint_layout
# ---------------------------------------------------------------------------

def test_layout_pretrained_when_checkpoint_has_only_the_base_blocks():
    assert dit_checkpoint_layout(set(range(12)), n_layers=24, n_pretrained=12) == "pretrained"


def test_layout_module_when_checkpoint_already_has_every_block():
    """The failing case from the bug report: a trained 24-layer checkpoint reloaded at
    eval time, with pretrained_dit_layers still 12."""
    assert dit_checkpoint_layout(set(range(24)), n_layers=24, n_pretrained=12) == "module"


def test_layout_module_when_no_extra_blocks_exist():
    """n_layers == n_pretrained: both shapes coincide; module (i.e. no remap) wins."""
    assert dit_checkpoint_layout(set(range(12)), n_layers=12, n_pretrained=12) == "module"


def test_layout_raises_for_a_shape_matching_neither():
    with pytest.raises(ValueError):
        dit_checkpoint_layout(set(range(13)), n_layers=24, n_pretrained=12)


# ---------------------------------------------------------------------------
# 2. zero_init_output_projections
# ---------------------------------------------------------------------------

def test_zero_init_makes_block_the_identity():
    torch.manual_seed(0)
    block = FlowBlock(dim=64, heads=4, use_cross_attn=True)
    zero_init_output_projections(block)

    cx = torch.randn(2, 5, 64)
    c = torch.randn(2, 64)
    context = torch.randn(2, 7, 64)
    block.eval()  # dropout must not perturb the (already exact) zero contributions
    # is_causal=False: not_causal avoids an unrelated, pre-existing bug in FlowerAttention
    # (passing both attn_mask and is_causal=True to F.scaled_dot_product_attention raises);
    # this test is about zero_init's identity property, not causal masking.
    out = block(cx, c, context=context, is_causal=False)
    assert torch.equal(out, cx)


def test_without_zero_init_block_is_not_identity():
    """Sanity check that the identity result above comes from zero_init, not from the
    block's forward pass being a no-op by construction."""
    torch.manual_seed(0)
    block = FlowBlock(dim=64, heads=4, use_cross_attn=True)
    cx = torch.randn(2, 5, 64)
    c = torch.randn(2, 64)
    context = torch.randn(2, 7, 64)
    block.eval()
    out = block(cx, c, context=context, is_causal=False)
    assert not torch.equal(out, cx)


# ---------------------------------------------------------------------------
# 3. FlowBlock width knobs
# ---------------------------------------------------------------------------

def test_lora_dim_controls_adaln_down_projection_width():
    block = FlowBlock(dim=1024, heads=16, lora_dim=128)
    assert block.adaLN_modulation[1].out_features == 128
    assert block.adaLN_modulation[2].in_features == 128
    assert block.adaLN_modulation[2].out_features == 6 * 1024


def test_mlp_hidden_dim_controls_swiglu_width():
    from flower.models.networks.transformers import find_multiple

    block = FlowBlock(dim=1024, heads=16, mlp_hidden_dim=4096)
    expected = find_multiple(int(2 * 4096 / 3), 256)
    assert block.mlp.fc1.out_features == expected


def test_mlp_hidden_dim_default_matches_current_behavior():
    block_default = FlowBlock(dim=1024, heads=16)
    block_explicit = FlowBlock(dim=1024, heads=16, mlp_hidden_dim=None)
    assert block_default.mlp.fc1.out_features == block_explicit.mlp.fc1.out_features == 2816


# ---------------------------------------------------------------------------
# 4. FLOWERVLA._get_param_groups
# ---------------------------------------------------------------------------

class _ParamGroupStub(torch.nn.Module):
    """Minimal stand-in exposing exactly what _get_param_groups reads: named_parameters()
    (real nn.Module machinery), self.optimizer_config, self.extra_dit_indices."""

    def __init__(self, optimizer_config, extra_dit_indices):
        super().__init__()
        self.vlm = torch.nn.Linear(4, 4)  # -> "vlm.weight" / "vlm.bias"
        self.dit = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(18)])  # -> "dit.{i}.weight/bias"
        self.cond_linear = torch.nn.Linear(4, 4, bias=False)  # pretrained-expert, no bias -> decay group
        self.optimizer_config = optimizer_config
        self.extra_dit_indices = extra_dit_indices


def _make_optimizer_config(learning_rate=2e-5, dit_learning_rate=None, new_layer_learning_rate=None):
    return OmegaConf.create({
        "transformer_weight_decay": 0.05,
        "learning_rate": learning_rate,
        "dit_learning_rate": dit_learning_rate,
        "new_layer_learning_rate": new_layer_learning_rate,
    })


def _all_params(groups):
    return [p for g in groups for p in g["params"]]


def test_param_groups_partition_all_trainable_params_exactly():
    stub = _ParamGroupStub(_make_optimizer_config(), extra_dit_indices={12, 13, 14, 15, 16, 17})
    groups = FLOWERVLA._get_param_groups(stub)

    all_params = _all_params(groups)
    expected = list(stub.parameters())
    assert len(all_params) == len(expected)
    assert {id(p) for p in all_params} == {id(p) for p in expected}
    # No parameter appears in two groups.
    ids = [id(p) for p in all_params]
    assert len(ids) == len(set(ids))


def test_default_config_uses_one_learning_rate_for_every_group():
    """dit_learning_rate/new_layer_learning_rate both null -> every group falls back to
    learning_rate, i.e. today's single-LR behavior."""
    stub = _ParamGroupStub(_make_optimizer_config(learning_rate=2e-5), extra_dit_indices={12, 13, 14, 15, 16, 17})
    groups = FLOWERVLA._get_param_groups(stub)
    assert all(g["lr"] == 2e-5 for g in groups)


def test_separate_learning_rates_route_to_the_right_family():
    stub = _ParamGroupStub(
        _make_optimizer_config(learning_rate=2e-5, dit_learning_rate=1e-4, new_layer_learning_rate=2e-4),
        extra_dit_indices={12, 13, 14, 15, 16, 17},
    )
    groups = FLOWERVLA._get_param_groups(stub)

    def lr_of(param):
        for g in groups:
            if any(p is param for p in g["params"]):
                return g["lr"]
        raise AssertionError("param not found in any group")

    assert lr_of(stub.vlm.weight) == 2e-5
    assert lr_of(stub.dit[0].weight) == 1e-4  # pretrained expert block
    assert lr_of(stub.cond_linear.weight) == 1e-4  # pretrained expert, non-DiT
    assert lr_of(stub.dit[12].weight) == 2e-4  # fresh expert block


def test_new_layer_lr_falls_back_to_dit_lr_when_unset():
    stub = _ParamGroupStub(
        _make_optimizer_config(learning_rate=2e-5, dit_learning_rate=1e-4, new_layer_learning_rate=None),
        extra_dit_indices={12, 13, 14, 15, 16, 17},
    )
    groups = FLOWERVLA._get_param_groups(stub)
    fresh_group = next(g for g in groups if any(p is stub.dit[12].weight for p in g["params"]))
    assert fresh_group["lr"] == 1e-4


# ---------------------------------------------------------------------------
# 5. TriStageLRScheduler group-aware stepping
# ---------------------------------------------------------------------------

def _scheduler_cfg(lr=2e-5):
    return OmegaConf.create({
        "lr_scheduler": {
            "init_lr": lr,
            "init_lr_scale": 0.1,
            "final_lr_scale": 0.5,
            "total_steps": 100,
            "phase_ratio": "(0.1, 0.4, 0.5)",
            "lr": lr,
        }
    })


def test_single_group_schedule_is_unaffected():
    """A single param group at peak_lr reproduces the pre-existing scalar LR trajectory.
    total_steps=100, phase_ratio=(0.1, 0.4, 0.5) -> warmup_steps=10, hold_steps=40; by
    step 15 the schedule is in its hold stage, at peak_lr."""
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([{"params": [param], "lr": 2e-5}])
    scheduler = TriStageLRScheduler(optimizer, _scheduler_cfg(lr=2e-5))

    lrs = [scheduler.step() for _ in range(15)]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(lrs[-1])
    # Non-trivial schedule: still warming up at step 0, at peak by step 15 (hold stage).
    assert lrs[0] < lrs[-1]
    assert lrs[-1] == pytest.approx(2e-5)


def test_two_groups_keep_a_constant_ratio_across_the_whole_schedule():
    """A 5x higher base LR for one group must stay ~5x the other's at every step, through
    warmup, hold, and decay."""
    p1 = torch.nn.Parameter(torch.zeros(1))
    p2 = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([
        {"params": [p1], "lr": 2e-5},
        {"params": [p2], "lr": 1e-4},
    ])
    scheduler = TriStageLRScheduler(optimizer, _scheduler_cfg(lr=2e-5))

    for step in range(100):
        scheduler.step()
        lr1 = optimizer.param_groups[0]["lr"]
        lr2 = optimizer.param_groups[1]["lr"]
        if lr1 != 0:
            assert lr2 / lr1 == pytest.approx(5.0), f"ratio drifted at step {step}"


# ---------------------------------------------------------------------------
# 6. FLOWERVLA._load_pretrained_weights -- checkpoint-layout gating
# ---------------------------------------------------------------------------

class _LoadWeightsStub(torch.nn.Module):
    """Minimal stand-in exposing exactly what _load_pretrained_weights reads/writes: a real
    ModuleList of FlowBlocks (so load_state_dict's key matching is real), plus the
    action-expert-capacity bookkeeping flower.py's _setup_dit_components computes."""

    def __init__(self, n_layers, pretrained_dit_layers, placement, extra_layer_init):
        super().__init__()
        self.device = "cpu"
        self.action_expert_from_scratch = False
        self.dit = torch.nn.ModuleList([FlowBlock(dim=64, heads=4) for _ in range(n_layers)])
        self.pretrained_dit_layers = pretrained_dit_layers
        self.extra_layer_init = extra_layer_init
        self.dit_layer_sources = dit_layer_sources(n_layers, pretrained_dit_layers, placement, extra_layer_init)
        # Positions are placement-derived, not init-derived -- mirrors flower.py's own
        # _setup_dit_components, which recomputes this with init="random" for the same reason.
        reference = dit_layer_sources(n_layers, pretrained_dit_layers, placement, "random")
        self.extra_dit_indices = {i for i, src in enumerate(reference) if src is None}


def _save_dit_checkpoint(path, blocks: dict) -> None:
    """A minimal (non-Lightning) checkpoint whose state_dict has dit.<idx>.<key> for each
    given block, nothing else -- enough for _load_pretrained_weights's DiT-remap path."""
    state_dict = {}
    for idx, block in blocks.items():
        for key, value in block.state_dict().items():
            state_dict[f"dit.{idx}.{key}"] = value
    torch.save({"state_dict": state_dict}, path)


def test_load_pretrained_weights_remaps_and_zero_inits_the_base_checkpoint(tmp_path):
    """A checkpoint carrying only the pretrained_dit_layers blocks (checkpoint-index space,
    e.g. the pretrained base loaded at train time) is remapped onto module slots, and the
    resulting fresh extra blocks are zero-inited."""
    stub = _LoadWeightsStub(n_layers=4, pretrained_dit_layers=2, placement="append", extra_layer_init="zero")
    checkpoint_blocks = {0: FlowBlock(dim=64, heads=4), 1: FlowBlock(dim=64, heads=4)}
    ckpt_path = tmp_path / "base.pt"
    _save_dit_checkpoint(ckpt_path, checkpoint_blocks)

    FLOWERVLA._load_pretrained_weights(stub, str(ckpt_path))

    assert torch.equal(stub.dit[0].mlp.proj.weight, checkpoint_blocks[0].mlp.proj.weight)
    assert torch.equal(stub.dit[1].mlp.proj.weight, checkpoint_blocks[1].mlp.proj.weight)
    # Extra blocks (module slots 2, 3) were fresh, not present in the checkpoint -- zeroed.
    for i in (2, 3):
        assert torch.all(stub.dit[i].mlp.proj.weight == 0)
        assert torch.all(stub.dit[i].self_attn.proj.weight == 0)


def test_load_pretrained_weights_loads_a_trained_checkpoint_verbatim(tmp_path):
    """Regression test for the bug this fixes: a checkpoint already covering every module
    slot (a finetuned/trained checkpoint reloaded at eval time -- see
    flower/evaluation/utils.py's load_mode_from_safetensor) must load as-is, with NO remap
    and NO zero-init of the "extra" positions -- those blocks were actually trained, unlike
    the base-checkpoint case above."""
    stub = _LoadWeightsStub(n_layers=4, pretrained_dit_layers=2, placement="append", extra_layer_init="zero")
    checkpoint_blocks = {i: FlowBlock(dim=64, heads=4) for i in range(4)}
    ckpt_path = tmp_path / "trained.pt"
    _save_dit_checkpoint(ckpt_path, checkpoint_blocks)

    FLOWERVLA._load_pretrained_weights(stub, str(ckpt_path))

    for i in range(4):
        assert torch.equal(stub.dit[i].mlp.proj.weight, checkpoint_blocks[i].mlp.proj.weight)
        assert torch.equal(stub.dit[i].self_attn.proj.weight, checkpoint_blocks[i].self_attn.proj.weight)


def test_load_pretrained_weights_rejects_an_unrecognized_layer_count(tmp_path):
    stub = _LoadWeightsStub(n_layers=4, pretrained_dit_layers=2, placement="append", extra_layer_init="zero")
    checkpoint_blocks = {i: FlowBlock(dim=64, heads=4) for i in range(3)}  # neither 2 nor 4 blocks
    ckpt_path = tmp_path / "malformed.pt"
    _save_dit_checkpoint(ckpt_path, checkpoint_blocks)

    with pytest.raises(ValueError):
        FLOWERVLA._load_pretrained_weights(stub, str(ckpt_path))
