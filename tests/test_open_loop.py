"""Tests for flower/evaluation/open_loop.py (open-loop per-timestep loss on held-out CALVIN
demonstrations) and scripts/open_loop_analysis.py's aggregation helpers.

- fm_loss_per_sample is rf_loss's formula, per sample and per action-chunk step, at a
  caller-chosen t and noise.
- Noise is keyed on the frame id, so every model/combo/batch layout sees the same draw.
- set_combo resets the eval mask for the all-on combo (apply_eval_modalities never does).
- gripper_phases / window_index annotate each sample with its time and phase in a window.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from flower.evaluation.open_loop import (
    annotation_batches,
    assert_single_instruction,
    fm_loss_per_sample,
    gripper_phases,
    sample_noise,
    sampled_mse_per_sample,
    set_combo,
    window_index,
)
from flower.models.flower import FLOWERVLA

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from open_loop_analysis import combo_label, paired_delta, timestep_bin  # noqa: E402


class _LossStub:
    """Exposes what rf_loss / sample_actions read from self, with a cheap fake DiT."""

    rf_loss = FLOWERVLA.rf_loss
    sample_actions = FLOWERVLA.sample_actions

    def __init__(self):
        self._param = torch.zeros(1)
        self.sampling_type = "uniform"
        self.num_sampling_steps = 4

    def parameters(self):
        yield self._param

    def dit_forward(self, z, t, cond):
        return 0.5 * z + t.view(-1, 1, 1) + cond["bias"]


def _cond(B):
    return {"bias": torch.linspace(-0.2, 0.2, B).view(B, 1, 1)}


def test_fm_loss_matches_rf_loss_formula_at_same_t_and_noise():
    model = _LossStub()
    actions = torch.rand(1, 10, 7) * 2 - 1
    cond = _cond(1)

    torch.manual_seed(0)
    expected, _ = model.rf_loss(cond, actions)
    # Replay rf_loss's draws: B=1 "uniform" t is rand(1) % (1 - eps), then z1 = randn_like.
    torch.manual_seed(0)
    t = float(torch.rand(1) % (1 - 1e-5))
    z1 = torch.randn_like(actions)

    got = fm_loss_per_sample(model, cond, actions, [t], z1.unsqueeze(1))
    assert got.shape == (1, 1, 10)
    assert torch.allclose(got.mean(), expected, atol=1e-6)


def test_fm_loss_shape_and_per_sample_independence():
    model = _LossStub()
    B, K = 3, 2
    actions = torch.rand(B, 10, 7)
    noise = torch.randn(B, K, 10, 7)
    full = fm_loss_per_sample(model, _cond(B), actions, [0.2, 0.8], noise)
    assert full.shape == (B, K, 10)
    # Sample 1 alone gives the same numbers as inside the batch.
    one = fm_loss_per_sample(model, {"bias": _cond(B)["bias"][1:2]}, actions[1:2], [0.2, 0.8], noise[1:2])
    assert torch.allclose(full[1:2], one)


def test_sampled_mse_per_sample_shape_and_value():
    model = _LossStub()
    actions = torch.zeros(2, 10, 7)
    noise = torch.randn(2, 10, 7)
    mse = sampled_mse_per_sample(model, _cond(2), actions, noise)
    pred = model.sample_actions(noise, _cond(2), inference=True)
    assert mse.shape == (2, 10)
    assert torch.allclose(mse, (pred**2).mean(-1))


def test_noise_keyed_on_frame_not_batch_position():
    a = sample_noise([5, 9, 11], n=3, shape=(10, 7), seed=0)
    b = sample_noise([11, 5], n=3, shape=(10, 7), seed=0)
    assert a.shape == (3, 3, 10, 7)
    assert torch.equal(a[0], b[1]) and torch.equal(a[2], b[0])
    assert not torch.equal(a[0], a[1])
    assert not torch.equal(a[0], sample_noise([5], n=3, shape=(10, 7), seed=1)[0])


def test_set_combo_resets_mask_for_all_on():
    model = types.SimpleNamespace(eval_modality_mask=None, eval_proprio_mask=False)
    set_combo(model, {"rgb_static": True, "rgb_gripper": False, "language": True, "proprio": True})
    assert model.eval_modality_mask == (True, False, True)
    assert model.eval_proprio_mask is None
    set_combo(model, {"rgb_static": True, "rgb_gripper": True, "language": True, "proprio": True})
    assert model.eval_modality_mask is None


def test_set_combo_rejects_empty_combo():
    model = types.SimpleNamespace(eval_modality_mask=None, eval_proprio_mask=None)
    with pytest.raises(ValueError):
        set_combo(model, {"rgb_static": False, "rgb_gripper": False, "language": False, "proprio": True})


def test_gripper_phases_never_closes():
    assert gripper_phases(np.ones(8)) == ["no_grasp"] * 8


def test_gripper_phases_close_then_reopen():
    # open x6, closed x8, open x6 (CALVIN: +1 open, -1 close)
    cmd = np.array([1] * 6 + [-1] * 8 + [1] * 6)
    phases = gripper_phases(cmd, margin=2)
    assert phases[:4] == ["approach"] * 4
    assert phases[4:9] == ["grasp"] * 5  # switch at 6, frames 4..8
    assert phases[9:12] == ["hold"] * 3
    assert phases[12:17] == ["release"] * 5  # switch at 14, frames 12..16
    assert phases[17:] == ["retreat"] * 3


def test_gripper_phases_starts_closed():
    cmd = np.array([-1] * 6 + [1] * 6)
    phases = gripper_phases(cmd, margin=1)
    assert phases[:5] == ["hold"] * 5
    assert phases[5:8] == ["release"] * 3
    assert phases[8:] == ["retreat"] * 4


def test_window_index_mirrors_dataset_lookup():
    # Two overlapping windows; min_window_size 3 -> starts s .. e+1-3.
    indx = [(10, 15), (12, 20)]
    tasks = ["open_drawer", "lift_red_block_table"]
    anns = ["open the drawer", "lift the red block"]
    gripper = {f: (1.0 if f < 16 else -1.0) for f in range(10, 21)}
    df = window_index(indx, tasks, anns, min_window_size=3, gripper_cmd=lambda fs: np.array([gripper[f] for f in fs]))

    # Same idx order as DiskDataset._build_file_indices_lang: range(s, e + 1 - min_window_size).
    assert list(df["idx"]) == list(range(3 + 6))
    assert list(df["frame"]) == list(range(10, 13)) + list(range(12, 18))
    assert list(df["ann_idx"]) == [0] * 3 + [1] * 6
    assert list(df["task"][:3]) == ["open_drawer"] * 3
    assert list(df["ann_text"][3:]) == ["lift the red block"] * 6
    assert list(df["offset"][3:]) == list(range(6))
    assert df.loc[df.ann_idx == 0, "progress"].tolist() == pytest.approx([0, 0.5, 1])
    assert df["progress"].between(0, 1).all()
    assert set(df.loc[df.ann_idx == 0, "phase"]) == {"no_grasp"}
    # Window 1 closes at frame 16: grasp spans frames 14..18 (margin 2), phases computed
    # over the whole window, read at each sample's observation frame.
    assert df.loc[(df.ann_idx == 1) & (df.frame == 12), "phase"].item() == "approach"
    assert df.loc[(df.ann_idx == 1) & (df.frame == 17), "phase"].item() == "grasp"


def test_annotation_batches_never_mix_windows():
    # Samples of one window share one instruction, so a batch never pads language.
    ann_idx = [0] * 5 + [1] * 2 + [2] * 4
    batches = annotation_batches(ann_idx, batch_size=3)
    assert batches == [[0, 1, 2], [3, 4], [5, 6], [7, 8, 9], [10]]


def test_assert_single_instruction_rejects_mixed_batch():
    # A mixed batch would pad the shorter instruction and shift its loss.
    assert_single_instruction(["open the drawer"] * 3)
    with pytest.raises(ValueError, match="instruction"):
        assert_single_instruction(["open the drawer", "lift the red block"])


def test_combo_label():
    assert combo_label({"use_static": True, "use_wrist": True, "use_lang": True}) == "static+wrist+lang"
    assert combo_label({"use_static": False, "use_wrist": True, "use_lang": False}) == "wrist"


def test_timestep_bin_edges():
    bins = timestep_bin(pd.Series([0, 4, 5, 9, 54]), width=5)
    assert list(bins) == [0, 0, 5, 5, 50]


def test_paired_delta_is_per_sample_difference_to_all_inputs():
    df = pd.DataFrame(
        {
            "run": ["r"] * 4,
            "idx": [0, 1, 0, 1],
            "combo": ["static+wrist+lang", "static+wrist+lang", "static", "static"],
            "mse_mean": [1.0, 2.0, 4.0, 3.0],
        }
    )
    d = paired_delta(df, "mse_mean")
    # All inputs stays in, as the zero reference line.
    assert d.loc[d.combo == "static+wrist+lang", "delta"].tolist() == [0.0, 0.0]
    assert d[d.combo == "static"].sort_values("idx")["delta"].tolist() == [3.0, 1.0]


def test_paired_delta_skips_runs_scored_on_all_inputs_only():
    df = pd.DataFrame({"run": ["base"], "idx": [0], "combo": ["static+wrist+lang"], "mse_mean": [1.0]})
    assert paired_delta(df, "mse_mean").empty


def test_analyze_renders_figures_and_summaries(tmp_path):
    from open_loop_analysis import COMBO_ORDER, analyze

    rng = np.random.default_rng(0)
    rows = []
    for run, dropout, keep in (("a", True, 0.5), ("b", False, None)):
        combos = COMBO_ORDER if dropout else COMBO_ORDER[:1]
        for combo in combos:
            for idx in range(250):  # >= MIN_BIN_COUNT, so every line is drawn
                rows.append(
                    {
                        "run": run, "model": f"m_{run}", "modality_dropout": dropout, "keep_fraction": keep,
                        "combo": combo, "idx": idx, "offset": idx % 10,
                        "phase": ["approach", "grasp", "hold", "no_grasp"][idx % 4],
                        "task": ["open_drawer", "push_red_block_left"][idx % 2],
                        "mse_mean": rng.random(), "fm_mean": rng.random(),
                        "mse_k0": rng.random(), "mse_k1": rng.random(),
                        "fm_t0.1": rng.random(), "fm_t0.9": rng.random(),
                    }
                )
    analyze(pd.DataFrame(rows), tmp_path, bin_width=50)

    for name in ("loss_vs_timestep_mse_mean", "delta_vs_timestep_fm_mean", "phase_mse_mean", "task_fm_mean"):
        assert (tmp_path / f"{name}.pdf").exists()
        assert (tmp_path / f"{name}.png").exists()
    summary = pd.read_csv(tmp_path / "summary_by_phase.csv")
    assert {"model", "combo", "phase", "mse_mean_mean", "mse_mean_delta_mean"} <= set(summary.columns)
    # Only the dropout model has deltas; the baseline contributes its all-inputs row only.
    assert set(summary.loc[summary.model == "m_b", "combo"]) == {"static+wrist+lang"}
    assert summary.loc[summary.model == "m_b", "mse_mean_delta_mean"].isna().all()
    assert "static+wrist+lang" in set(summary.loc[summary.model == "m_a", "combo"])
