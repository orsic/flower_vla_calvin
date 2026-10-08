"""Tests for scripts/calvin_pipeline.py -- deciding the CALVIN evaluation a training run
needs (one eval-calvin line, or one per modality combo for a dropout run) and resolving
its result.csv for upload. Pure logic / filesystem, mirrors tests/test_mimicgen_pipeline.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from calvin_pipeline import csv_path, plan_lines  # noqa: E402

from flower.evaluation.eval_records import result_dir  # noqa: E402


def _write_train_cfg(train_folder: Path, *, modality_dropout=False, use_proprio=False):
    hydra_dir = train_folder / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    (hydra_dir / "config.yaml").write_text(
        f"""
benchmark_name: calvin_d
seed: 42
model:
  use_proprio: {str(use_proprio).lower()}
  modality_dropout: {str(modality_dropout).lower()}
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
