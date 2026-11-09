"""Tests for scripts/calvin_pipeline.py -- deciding the CALVIN evaluation a training run
needs (one eval-calvin line, or one per modality combo for a dropout run) and resolving
its result.csv for upload. Pure logic / filesystem, mirrors tests/test_mimicgen_pipeline.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from calvin_pipeline import csv_path, plan_lines  # noqa: E402

from flower.evaluation.eval_records import result_dir  # noqa: E402


def _write_train_cfg(train_folder: Path, *, modality_dropout=False, use_proprio=False, modalities=None):
    hydra_dir = train_folder / ".hydra"
    mods = "".join(f"\n    {k}: {str(v).lower()}" for k, v in (modalities or {}).items()) or " {}"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    (hydra_dir / "config.yaml").write_text(
        f"""
benchmark_name: calvin_d
seed: 42
model:
  use_proprio: {str(use_proprio).lower()}
  modality_dropout: {str(modality_dropout).lower()}
  modalities:{mods}
logger:
  project: multimodal_florence
  entity: some-entity
"""
    )


def test_plan_lines_baseline_single_eval_calvin_line(tmp_path):
    train_folder = tmp_path / "calvin_abcd_d" / "run1"
    _write_train_cfg(train_folder)

    plan = plan_lines(str(train_folder), [], reeval=False)

    assert len(plan) == 1
    service, overrides = plan[0]
    assert service == "eval-calvin"
    assert f"train_folder={train_folder}" in overrides
    assert f"checkpoint={train_folder / 'seed_42' / 'saved_models' / 'last.ckpt'}" in overrides
    assert "num_videos=0" in overrides
    assert not any(o.startswith("eval_modalities.") for o in overrides)
    assert "reeval=True" not in overrides


def test_plan_lines_dropout_one_line_per_combo_reeval_first_only(tmp_path):
    train_folder = tmp_path / "calvin_abcd_d_dropout" / "run1"
    _write_train_cfg(train_folder, modality_dropout=True)

    plan = plan_lines(str(train_folder), ["num_sequences=8"], reeval=True)

    assert len(plan) == 7
    assert {service for service, _ in plan} == {"eval-calvin"}
    combos = {tuple(o for o in overrides if o.startswith("eval_modalities.")) for _, overrides in plan}
    assert len(combos) == 7
    assert all("num_sequences=8" in overrides for _, overrides in plan)
    assert ["reeval=True" in overrides for _, overrides in plan] == [True] + [False] * 6


def test_csv_path_is_calvin_suite_result(tmp_path):
    train_folder = tmp_path / "calvin_abcd_d" / "run1"
    _write_train_cfg(train_folder)
    checkpoint = train_folder / "seed_42" / "saved_models" / "last.ckpt"

    assert csv_path(str(train_folder), []) == result_dir(train_folder, checkpoint, "calvin", "ABCD_D") / "result.csv"


def test_plan_lines_fixed_subset_run_names_its_trained_inputs(tmp_path):
    """A no-dropout run trained without an input is evaluated with exactly its trained
    inputs, stated explicitly so result.csv's use_* columns record them."""
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, use_proprio=True, modalities={"rgb_static": False})
    plan = plan_lines(str(train_folder), [], reeval=False)
    assert len(plan) == 1
    overrides = plan[0][1]
    assert "eval_modalities.rgb_static=False" in overrides
    assert "eval_modalities.rgb_gripper=True" in overrides
    assert "eval_modalities.language=True" in overrides
    assert "eval_modalities.proprio=True" in overrides


def _flags(overrides):
    flags = dict(o.split("=", 1) for o in overrides if o.startswith("eval_modalities."))
    return tuple(flags[f"eval_modalities.{k}"] == "True" for k in ("rgb_static", "rgb_gripper", "language", "proprio"))


def test_plan_lines_all_combos_sweeps_a_non_dropout_run(tmp_path):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, use_proprio=True)
    plan = plan_lines(str(train_folder), [], reeval=True, all_combos=True)
    assert len(plan) == 14
    assert _flags(plan[0][1]) == (True, True, True, True)
    assert ["reeval=True" in o for _, o in plan] == [True] + [False] * 13


def test_plan_lines_all_combos_with_skip_multi_off_within_trained_inputs(tmp_path):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, use_proprio=True, modalities={"language": False})
    plan = plan_lines(str(train_folder), [], reeval=False, all_combos=True, skip_multi_off=True)
    assert {_flags(o) for _, o in plan} == {
        (True, True, False, True),
        (False, True, False, True),
        (True, False, False, True),
        (True, True, False, False),
    }


def test_plan_lines_resume_skips_combos_in_result_csv(tmp_path):
    from flower.evaluation.eval_records import write_csv

    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, use_proprio=True)
    row = {"use_rgb_static": 1, "use_rgb_gripper": 1, "use_language": 1, "use_proprio": 1, "success": 1}
    write_csv(csv_path(str(train_folder), []), [row])
    plan = plan_lines(str(train_folder), [], reeval=False, all_combos=True, skip_multi_off=True, resume=True)
    assert len(plan) == 4
    assert (True, True, True, True) not in {_flags(o) for _, o in plan}
    # nothing left: an empty plan
    assert plan_lines(str(train_folder), [], reeval=False, resume=True) == []


def test_restore_puts_calvin_csv_back(tmp_path, monkeypatch):
    import calvin_pipeline
    from flower.evaluation.eval_records import read_csv, write_csv

    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, use_proprio=True)
    art = tmp_path / "artifact"
    write_csv(art / "calvin.csv", [{"use_rgb_static": 1, "success": 1}])
    monkeypatch.setattr(calvin_pipeline, "download_eval_artifact", lambda *a, **k: art)
    calvin_pipeline.restore(str(train_folder), [])
    assert len(read_csv(csv_path(str(train_folder), []))) == 1
