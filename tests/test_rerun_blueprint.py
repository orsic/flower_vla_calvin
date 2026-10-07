"""Tests for scripts/rerun_blueprint.py -- the .rbl viewer layout for rollout recordings.

The blueprint is generated from the combos actually present in a recording, so these
drive a real RerunRecorder to produce one and then check the blueprint names the rows it
found. No MuJoCo, no model.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from flower.evaluation.rerun_recorder import RerunRecorder  # noqa: E402

rr = pytest.importorskip("rerun")
blueprint = pytest.importorskip("rerun_blueprint")

ALL_ON = {"rgb_static": True, "rgb_gripper": True, "language": True, "proprio": True}


def _record(path, eval_modalities):
    rec = RerunRecorder(
        {"enabled": True, "max_episodes": 2, "path": str(path), "frame_stride": 50,
         "recording_id": "test"},
        str(path.parent), eval_modalities, True,
    )
    rec.begin_batch([("task_a", 0)], "do the thing")
    obs = [{
        "agentview_image": np.zeros((8, 8, 3), np.uint8),
        "robot0_eye_in_hand_image": np.zeros((8, 8, 3), np.uint8),
        "robot0_joint_pos": np.zeros(7),
        "robot0_gripper_qpos": np.zeros(2),
    }]
    rec.log_step(1, obs, np.zeros((1, 7)), [0])
    rec.end_batch([True], [1], [0])
    rec.close()
    return rec


def test_combos_are_read_back_from_the_recordings(tmp_path):
    """The blueprint only ever names a row that has data behind it."""
    a, b = tmp_path / "a.rrd", tmp_path / "b.rrd"
    rec_a = _record(a, ALL_ON)
    rec_b = _record(b, {**ALL_ON, "rgb_gripper": False})

    roots = blueprint.combos_in([a, b])
    assert set(roots) == {rec_a.root, rec_b.root}
    # Full-modality arm first, so it is the top row.
    assert roots[0] == rec_a.root


def test_single_recording_yields_one_row(tmp_path):
    a = tmp_path / "a.rrd"
    rec = _record(a, ALL_ON)
    assert blueprint.combos_in([a]) == [rec.root]


def test_row_uses_the_requested_layout(tmp_path):
    """Two images then a vertical stack of action over proprio, all side by side."""
    row = blueprint.arm_row("/static\\+lang")
    kinds = [type(c).__name__ for c in row.contents]
    assert kinds == ["Spatial2DView", "Spatial2DView", "Vertical"]
    origins = [str(c.origin) for c in row.contents[:2]]
    assert origins == ["/static\\+lang/input/rgb_static", "/static\\+lang/input/rgb_gripper"]
    charts = [str(c.origin) for c in row.contents[2].contents]
    assert charts == ["/static\\+lang/output/action", "/static\\+lang/input/proprio"]


def test_blueprint_saves_a_loadable_rbl(tmp_path):
    a, b = tmp_path / "a.rrd", tmp_path / "b.rrd"
    _record(a, ALL_ON)
    _record(b, {**ALL_ON, "rgb_gripper": False})
    out = tmp_path / "layout.rbl"

    blueprint.main([str(out), str(a), str(b)])

    # A .rbl holds a blueprint store, not a recording, so load_recording can't read it
    # back -- check it is a well-formed Rerun archive instead.
    assert out.read_bytes()[:4] == b"RRF2"
    assert out.stat().st_size > 0


def test_explicit_combo_flag_needs_no_recording(tmp_path):
    out = tmp_path / "layout.rbl"
    blueprint.main([str(out), "--combo", "static+wrist+lang+proprio", "--combo", "static+lang"])
    assert out.exists()


def test_no_combos_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        blueprint.main([str(tmp_path / "x.rbl")])
