"""Tests for flower.evaluation.rerun_recorder -- Rerun .rrd rollout recordings.

Pure logic tests: synthetic observation dicts and action arrays, no MuJoCo, no model.
Assertions read the written .rrd back through rerun's dataframe API, so they check what
actually landed in the file rather than what the recorder was asked to log.
"""

import numpy as np
import pytest

from flower.evaluation.rerun_recorder import (
    ACTION_NAMES,
    PROPRIO_NAMES,
    TIMELINE,
    RerunRecorder,
    combo_entity_path,
    modality_flags,
    modality_label,
)

rr = pytest.importorskip("rerun")

ALL_ON = {"rgb_static": True, "rgb_gripper": True, "language": True, "proprio": True}


def _cfg(path, **overrides):
    cfg = {
        "enabled": True,
        "max_episodes": 10,
        "jpeg_quality": 40,
        "path": str(path),
        "combo_name": None,
        "frame_stride": 100,
        "recording_id": "test",
    }
    cfg.update(overrides)
    return cfg


def _obs(seed=0, with_proprio=True, with_wrist=True):
    rng = np.random.RandomState(seed)
    obs = {"agentview_image": rng.randint(0, 255, (16, 16, 3), dtype=np.uint8)}
    if with_wrist:
        obs["robot0_eye_in_hand_image"] = rng.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    if with_proprio:
        obs["robot0_joint_pos"] = rng.rand(7)
        obs["robot0_gripper_qpos"] = rng.rand(2)
    return obs


def _run_episodes(recorder, keys, languages, n_steps=3, active=None, dones=None):
    """Drive one batch through the recorder's begin/log/end call sequence."""
    b = len(keys)
    recorder.begin_batch(keys, languages)
    obs = [_obs(seed=k) for k in range(b)]
    for step in range(1, n_steps + 1):
        ids = list(range(b)) if active is None else active(step)
        recorder.log_step(step, obs, np.full((b, 7), 0.1 * step), ids)
    recorder.end_batch(
        dones if dones is not None else [True] * b, [n_steps] * b, list(range(b))
    )


def _entities(path):
    rec = rr.dataframe.load_recording(str(path))
    return {c.entity_path for c in rec.schema().component_columns()}


def _table(path, entity, component="Scalars:scalars"):
    rec = rr.dataframe.load_recording(str(path))
    return rec.view(index=TIMELINE, contents={entity: [component]}).select().read_all()


def _rows(path, entity, component="Scalars:scalars"):
    return _table(path, entity, component).num_rows


def _frames(path, entity, component="Scalars:scalars"):
    """The frame index of every row logged to an entity."""
    t = _table(path, entity, component)
    return [v.as_py() for v in t.column(TIMELINE)]


def _static_free_values(path, entity, component):
    """Values of one time-logged AnyValues field, in frame order."""
    t = _table(path, entity, component)
    return [v.as_py()[0] for v in t.column(f"{entity}:{component}") if v is not None]


# --- 1. disabled ------------------------------------------------------------------


def test_disabled_recorder_is_a_noop(tmp_path):
    out = tmp_path / "rollouts.rrd"
    rec = RerunRecorder(_cfg(out, enabled=False), str(tmp_path), ALL_ON, True)
    assert not rec.enabled
    _run_episodes(rec, [("task_a", 0)], "pick it up")
    rec.close()
    assert not out.exists()


def test_max_episodes_zero_disables_recording(tmp_path):
    out = tmp_path / "rollouts.rrd"
    rec = RerunRecorder(_cfg(out, max_episodes=0), str(tmp_path), ALL_ON, True)
    assert not rec.enabled
    rec.close()
    assert not out.exists()


# --- 2. full recording ------------------------------------------------------------


def test_records_images_actions_proprio_and_success(tmp_path):
    out = tmp_path / "rollouts.rrd"
    rec = RerunRecorder(_cfg(out), str(tmp_path), ALL_ON, True)
    assert rec.enabled
    _run_episodes(rec, [("task_a", 0), ("task_a", 1)], "pick up the bowl", n_steps=3)
    rec.close()

    assert out.exists() and out.stat().st_size > 0
    entities = _entities(out)
    base = rec.root
    for leaf in ("input/rgb_static", "input/rgb_gripper", "input/proprio",
                 "input/present", "output/action", "output/success", "info", "summary"):
        assert f"{base}/{leaf}" in entities, leaf

    # Two episodes x 3 steps, all under the one set of paths.
    assert _rows(out, f"{base}/output/action") == 6
    assert _rows(out, f"{base}/input/proprio") == 6
    assert _rows(out, f"{base}/output/success") == 2


def test_action_and_proprio_values_round_trip(tmp_path):
    out = tmp_path / "rollouts.rrd"
    rec = RerunRecorder(_cfg(out), str(tmp_path), ALL_ON, True)
    rec.begin_batch([("task_a", 0)], "do the thing")
    obs = [_obs(seed=0)]
    rec.log_step(1, obs, np.arange(7, dtype=float).reshape(1, 7), [0])
    rec.end_batch([True], [1], [123])
    rec.close()

    base = rec.root
    table = (
        rr.dataframe.load_recording(str(out))
        .view(index=TIMELINE, contents={f"{base}/output/action": ["Scalars:scalars"]})
        .select()
        .read_all()
    )
    logged = table.column(f"{base}/output/action:Scalars:scalars")[0].as_py()
    assert logged == pytest.approx(list(range(7)))
    assert len(ACTION_NAMES) == 7 and len(PROPRIO_NAMES) == 9


def test_camera_frames_are_flipped_for_display_only(tmp_path):
    """robosuite renders bottom-up, so a raw agentview/wrist frame is upside down to a
    human. The recording flips it; the model is still fed the unflipped array (nothing
    here touches the inference path). Asserting it explicitly because the recorded image
    is therefore NOT byte-identical to the model input."""
    import cv2

    out = tmp_path / "rollouts.rrd"
    rec = RerunRecorder(_cfg(out), str(tmp_path), ALL_ON, True)
    # Top row white, everything else black -- unambiguous under a vertical flip.
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[0, :, :] = 255
    rec.begin_batch([("task_a", 0)], "do the thing")
    rec.log_step(1, [{"agentview_image": frame}], np.zeros((1, 7)), [0])
    rec.end_batch([True], [1], [0])
    rec.close()

    entity = f"{rec.root}/input/rgb_static"
    table = (
        rr.dataframe.load_recording(str(out))
        .view(index=TIMELINE, contents={entity: ["EncodedImage:blob"]})
        .select()
        .read_all()
    )
    blob = bytes(table.column(f"{entity}:EncodedImage:blob")[0].as_py()[0])
    decoded = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert decoded[-1].mean() > 200, "white row should have moved to the bottom"
    assert decoded[0].mean() < 55, "top row should now be the black one"


# --- 3. withheld modalities -------------------------------------------------------


def test_withheld_modality_is_still_logged_with_fed_zero(tmp_path):
    """eval_modalities.rgb_gripper=False withholds the wrist view from the model, but the
    env still renders it -- the frames stay in the recording and the drop shows up as a
    zero in the fed series, so the withheld arm is still watchable."""
    out = tmp_path / "rollouts.rrd"
    modalities = {**ALL_ON, "rgb_gripper": False}
    rec = RerunRecorder(_cfg(out), str(tmp_path), modalities, True)
    assert rec.label == "static+lang+proprio"
    _run_episodes(rec, [("task_a", 0)], "pick up the bowl", n_steps=2)
    rec.close()

    base = rec.root
    entities = _entities(out)
    assert f"{base}/input/rgb_gripper" in entities
    assert f"{base}/input/rgb_static" in entities

    table = (
        rr.dataframe.load_recording(str(out))
        .view(index=TIMELINE, contents={f"{base}/input/present": ["Scalars:scalars"]})
        .select()
        .read_all()
    )
    present = table.column(f"{base}/input/present:Scalars:scalars")[0].as_py()
    assert present == [1.0, 0.0, 1.0, 1.0]  # static, wrist, lang, proprio


def test_uses_proprio_false_marks_proprio_not_fed(tmp_path):
    """A use_proprio=False checkpoint never receives proprio however eval_modalities is
    set -- same rule as result.csv's use_proprio column."""
    out = tmp_path / "rollouts.rrd"
    rec = RerunRecorder(_cfg(out), str(tmp_path), ALL_ON, uses_proprio=False)
    assert rec.label == "static+wrist+lang"
    _run_episodes(rec, [("task_a", 0)], "pick up the bowl", n_steps=2)
    rec.close()

    base = rec.root
    # The env still reports proprio, so it is still recorded ...
    assert f"{base}/input/proprio" in _entities(out)
    # ... but flagged as not reaching the model.
    table = (
        rr.dataframe.load_recording(str(out))
        .view(index=TIMELINE, contents={f"{base}/input/present": ["Scalars:scalars"]})
        .select()
        .read_all()
    )
    assert table.column(f"{base}/input/present:Scalars:scalars")[0].as_py()[3] == 0.0


def test_obs_without_proprio_or_wrist_keys_does_not_crash(tmp_path):
    out = tmp_path / "rollouts.rrd"
    rec = RerunRecorder(_cfg(out), str(tmp_path), ALL_ON, True)
    rec.begin_batch([("task_a", 0)], "do the thing")
    rec.log_step(1, [_obs(with_proprio=False, with_wrist=False)], np.zeros((1, 7)), [0])
    rec.end_batch([False], [1], [7])
    rec.close()

    entities = _entities(out)
    base = rec.root
    assert f"{base}/input/rgb_static" in entities
    assert f"{base}/input/rgb_gripper" not in entities
    assert f"{base}/input/proprio" not in entities


def test_combo_root_defaults_to_modality_combo_and_is_overridable(tmp_path):
    out = tmp_path / "rollouts.rrd"
    rec = RerunRecorder(_cfg(out, combo_name="arm_b"), str(tmp_path), ALL_ON, True)
    _run_episodes(rec, [("task_a", 0)], "do the thing", n_steps=1)
    rec.close()
    assert rec.root == "/arm_b"
    assert "/arm_b/output/action" in _entities(out)


def test_entity_path_escapes_the_modality_separator():
    """'+' is not a bare entity-path character, so the combo root carries escapes on the
    wire (the viewer renders them back as '+'). A blueprint origin and a dataframe query
    both have to spell them, which is why callers go through combo_entity_path."""
    assert combo_entity_path("static+lang") == "/static\\+lang"


def test_modality_helpers():
    assert modality_label(modality_flags(ALL_ON, True)) == "static+wrist+lang+proprio"
    assert modality_label(modality_flags({"rgb_static": False}, False)) == "wrist+lang"
    assert modality_label({name: False for name in ("static", "wrist", "lang", "proprio")}) == "none"


# --- 4. episode selection and slot lifecycle --------------------------------------


def test_max_episodes_caps_episodes_across_batches(tmp_path):
    """The cap counts episodes in work order across every batch, so two evals that differ
    only in eval_modalities record the identical episode set into the identical slots."""
    out = tmp_path / "rollouts.rrd"
    rec = RerunRecorder(_cfg(out, max_episodes=5), str(tmp_path), ALL_ON, True)
    for batch in range(3):
        keys = [(f"task_{batch}", ep) for ep in range(4)]
        _run_episodes(rec, keys, "do the thing", n_steps=1)
    rec.close()

    # One info record per recorded episode, each at its own slot's first frame.
    frames = _frames(out, f"{rec.root}/info", "episode_slot")
    assert frames == [rec.episode_frame(slot) for slot in range(5)]

    recorded = _static_free_values(out, f"{rec.root}/info", "task_name")
    assert recorded == ["task_0"] * 4 + ["task_1"]


def test_episodes_occupy_fixed_frame_slots(tmp_path):
    """Episode i lives in [i * frame_stride, (i+1) * frame_stride) regardless of how
    long it ran -- that fixed width is what keeps two arms episode-aligned."""
    out = tmp_path / "rollouts.rrd"
    rec = RerunRecorder(_cfg(out, frame_stride=100), str(tmp_path), ALL_ON, True)
    _run_episodes(rec, [("task_a", 0), ("task_a", 1)], "do it", n_steps=3)
    _run_episodes(rec, [("task_a", 2)], "do it", n_steps=2)
    rec.close()

    stride = rec.frame_stride
    assert stride == 101  # frame_stride + 1, so step `frame_stride` still fits its slot
    # Slots 0 and 1 ran 3 steps, slot 2 ran 2.
    assert _frames(out, f"{rec.root}/output/action") == sorted(
        [1, 2, 3]
        + [stride + 1, stride + 2, stride + 3]
        + [2 * stride + 1, 2 * stride + 2]
    )


def test_two_arms_put_the_same_episode_in_the_same_slot(tmp_path):
    """The point of fixed slots: a full-modality arm and a withheld-modality arm whose
    episodes end at different steps still line up episode-for-episode, so one time
    cursor shows the same rollout in both."""
    a = tmp_path / "a.rrd"
    b = tmp_path / "b.rrd"
    rec_a = RerunRecorder(_cfg(a, frame_stride=100), str(tmp_path), ALL_ON, True)
    rec_b = RerunRecorder(
        _cfg(b, frame_stride=100), str(tmp_path), {**ALL_ON, "rgb_gripper": False}, True
    )
    keys = [("task_a", 0), ("task_a", 1)]
    _run_episodes(rec_a, keys, "do it", n_steps=2)   # arm A finishes early
    _run_episodes(rec_b, keys, "do it", n_steps=5)   # arm B runs longer
    rec_a.close()
    rec_b.close()

    assert rec_a.root != rec_b.root  # different combos -> sibling branches
    for rec, out in ((rec_a, a), (rec_b, b)):
        assert _frames(out, f"{rec.root}/info", "episode_slot") == [
            rec.episode_frame(0),
            rec.episode_frame(1),
        ]


def test_finished_slot_stops_being_logged(tmp_path):
    """A slot dropped from active_ids (its episode finished) gains no further rows, even
    though inference keeps running at full batch width."""
    out = tmp_path / "rollouts.rrd"
    rec = RerunRecorder(_cfg(out, frame_stride=100), str(tmp_path), ALL_ON, True)
    # Slot 0 finishes after step 1; slot 1 runs all 3 steps.
    _run_episodes(
        rec,
        [("task_a", 0), ("task_a", 1)],
        "do the thing",
        n_steps=3,
        active=lambda step: [0, 1] if step == 1 else [1],
    )
    rec.close()

    stride = rec.frame_stride
    frames = _frames(out, f"{rec.root}/output/action")
    assert [f for f in frames if f < stride] == [1]              # slot 0: one step only
    assert [f - stride for f in frames if f >= stride] == [1, 2, 3]  # slot 1: all three


def test_per_slot_languages_and_outcomes_are_recorded(tmp_path):
    """Cross-task batching gives one instruction per slot rather than one broadcast, and
    each episode's identity lands at its own slot's first frame so that scrubbing
    anywhere inside the episode resolves to the right record."""
    out = tmp_path / "rollouts.rrd"
    rec = RerunRecorder(_cfg(out, frame_stride=100), str(tmp_path), ALL_ON, True)
    _run_episodes(
        rec,
        [("task_a", 0), ("task_b", 0)],
        ["open the drawer", "turn on the stove"],
        n_steps=2,
        dones=[True, False],
    )
    rec.close()

    info = f"{rec.root}/info"
    assert _static_free_values(out, info, "language") == [
        "open the drawer",
        "turn on the stove",
    ]
    assert _static_free_values(out, info, "task_name") == ["task_a", "task_b"]
    assert _static_free_values(out, info, "success") == [1, 0]
    assert _frames(out, info, "language") == [rec.episode_frame(0), rec.episode_frame(1)]
    assert f"{rec.root}/summary" in _entities(out)
