"""Tests for scripts/eval_pipeline.py -- deciding which LIBERO/LIBERO-Plus evaluations a
training run needs, and resuming a partially-completed sweep.

Pure logic / filesystem tests, no MuJoCo, no model, no wandb network calls.
"""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from eval_pipeline import (  # noqa: E402
    artifact_name,
    full_modality_combo,
    modality_combos,
    parse_overrides,
    plan_lines,
    token_combos,
    wandb_run_id,
)

from flower.evaluation.eval_records import write_csv  # noqa: E402


def _write_train_cfg(train_folder: Path, *, dropout: bool, use_proprio: bool, benchmark="libero_10", seed=42):
    hydra_dir = train_folder / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    (hydra_dir / "config.yaml").write_text(
        f"""
libero_benchmark: {benchmark}
seed: {seed}
model:
  use_proprio: {use_proprio}
  modality_dropout: {dropout}
logger:
  project: multimodal_policies
  entity: some-entity
"""
    )


def _row(rgb_static, rgb_gripper, language, proprio):
    return {
        "use_rgb_static": int(rgb_static),
        "use_rgb_gripper": int(rgb_gripper),
        "use_language": int(language),
        "use_proprio": int(proprio),
        "task_name": "t0",
        "task_idx": 0,
        "episode_idx": 0,
        "checkpoint_name": "last",
        "success": 1,
    }


# ---------------------------------------------------------------------------
# token_combos / modality_combos
# ---------------------------------------------------------------------------


def test_token_combos_is_seven_all_on_first():
    combos = token_combos()
    assert len(combos) == 7
    assert combos[0] == {"rgb_static": True, "rgb_gripper": True, "language": True}
    # every combo has at least one modality on
    assert all(any(combo.values()) for combo in combos)
    # all distinct
    assert len({tuple(sorted(c.items())) for c in combos}) == 7


def test_modality_combos_without_proprio_is_seven_proprio_always_true():
    combos = modality_combos(use_proprio=False)
    assert len(combos) == 7
    assert all(c["proprio"] is True for c in combos)


def test_modality_combos_with_proprio_is_fourteen_no_proprio_only():
    combos = modality_combos(use_proprio=True)
    assert len(combos) == 14
    # "proprio only" (all three token modalities off) is never emitted
    assert not any(
        not c["rgb_static"] and not c["rgb_gripper"] and not c["language"] for c in combos
    )
    # each of the 7 token combos appears exactly twice (proprio True and False)
    token_only = [{"rgb_static": c["rgb_static"], "rgb_gripper": c["rgb_gripper"], "language": c["language"]} for c in combos]
    for combo in token_combos():
        assert token_only.count(combo) == 2


# ---------------------------------------------------------------------------
# parse_overrides
# ---------------------------------------------------------------------------


def test_parse_overrides_last_wins():
    parsed = parse_overrides(["benchmark_name=libero_10", "benchmark_name=libero_spatial"])
    assert parsed == {"benchmark_name": "libero_spatial"}


def test_parse_overrides_ignores_flags_without_equals():
    parsed = parse_overrides(["--resume", "n_eval=5"])
    assert parsed == {"n_eval": "5"}


# ---------------------------------------------------------------------------
# plan_lines
# ---------------------------------------------------------------------------


def test_plan_lines_regular_run_is_eval_then_eval_plus(tmp_path):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=False, use_proprio=False)

    lines = plan_lines(str(train_folder), resume=False, extra_overrides=[])

    assert [svc for svc, _ in lines] == ["eval", "eval-plus"]


def test_plan_lines_dropout_no_proprio_is_seven_plus_one(tmp_path):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=True, use_proprio=False)

    lines = plan_lines(str(train_folder), resume=False, extra_overrides=[])

    services = [svc for svc, _ in lines]
    assert services.count("eval") == 7
    assert services.count("eval-plus") == 1
    assert services[-1] == "eval-plus"


def test_plan_lines_dropout_with_proprio_is_fourteen_plus_one(tmp_path):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=True, use_proprio=True)

    lines = plan_lines(str(train_folder), resume=False, extra_overrides=[])

    services = [svc for svc, _ in lines]
    assert services.count("eval") == 14
    assert services.count("eval-plus") == 1


def test_plan_lines_default_checkpoint_and_train_folder(tmp_path):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=False, use_proprio=False, seed=7)

    _, overrides = plan_lines(str(train_folder), resume=False, extra_overrides=[])[0]

    assert f"train_folder={train_folder}" in overrides
    assert f"checkpoint={train_folder / 'seed_7' / 'saved_models' / 'last.ckpt'}" in overrides
    assert "benchmark_name=libero_10" in overrides


def test_plan_lines_overrides_appended_last_on_every_line(tmp_path):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=False, use_proprio=False)

    lines = plan_lines(str(train_folder), resume=False, extra_overrides=["n_eval=5", "eval_batch_size=32"])

    for _, overrides in lines:
        assert overrides[-2:] == ["n_eval=5", "eval_batch_size=32"]


def test_plan_lines_benchmark_name_override_propagates(tmp_path):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=False, use_proprio=False, benchmark="libero_10")

    _, overrides = plan_lines(
        str(train_folder), resume=False, extra_overrides=["benchmark_name=libero_spatial"]
    )[0]

    assert "benchmark_name=libero_spatial" in overrides
    assert "benchmark_name=libero_10" not in overrides


def test_plan_lines_resume_skips_already_evaluated_combo(tmp_path):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=False, use_proprio=True)
    csv_path = train_folder / "eval_logs" / "last" / "orig_libero_10" / "result.csv"
    write_csv(csv_path, [_row(True, True, True, True)])

    lines = plan_lines(str(train_folder), resume=True, extra_overrides=[])

    # The one combo a non-dropout run needs on LIBERO is already in result.csv; only
    # eval-plus (untouched) remains.
    assert [svc for svc, _ in lines] == ["eval-plus"]


def test_plan_lines_resume_keeps_uncovered_combo(tmp_path):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=True, use_proprio=True)
    csv_path = train_folder / "eval_logs" / "last" / "orig_libero_10" / "result.csv"
    # Only the all-on combo has been evaluated; the other 13 dropout combos have not.
    write_csv(csv_path, [_row(True, True, True, True)])

    lines = plan_lines(str(train_folder), resume=True, extra_overrides=[])

    assert [svc for svc, _ in lines].count("eval") == 13
    assert [svc for svc, _ in lines].count("eval-plus") == 1


def test_plan_lines_resume_accounts_for_no_proprio_model(tmp_path):
    """modality_combos() always sets proprio=True for a no-proprio model (there's nothing
    to sweep), but flower_eval_libero.py records what the model actually received -- so a
    real eval of the all-on combo writes use_proprio=0, not 1. Resume must match on that
    effective value or it would never mark a no-proprio combo done."""
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=True, use_proprio=False)
    csv_path = train_folder / "eval_logs" / "last" / "orig_libero_10" / "result.csv"
    write_csv(csv_path, [_row(True, True, True, False)])  # use_proprio=0, as a real run would write

    lines = plan_lines(str(train_folder), resume=True, extra_overrides=[])

    assert [svc for svc, _ in lines].count("eval") == 6
    assert [svc for svc, _ in lines].count("eval-plus") == 1


# ---------------------------------------------------------------------------
# wandb_run_id
# ---------------------------------------------------------------------------


def test_wandb_run_id_from_run_dir(tmp_path):
    run_dir = tmp_path / "libero_10_dropout" / "2026-08-28_15-45-43"
    run_dir.mkdir(parents=True)
    assert wandb_run_id(str(run_dir)) == "libero_10_dropout_2026-08-28_15-45-43"


# ---------------------------------------------------------------------------
# artifact_name -- W&B artifact names disallow the '+' a modality-ablation run.sh
# label (run.sh's modality_label) puts into the run id
# ---------------------------------------------------------------------------


def test_artifact_name_sanitizes_modality_ablation_plus():
    run_id = "libero_10_static+wrist+lang+proprio_2026-09-09_13-56-20"
    assert artifact_name(run_id) == "eval-libero_10_static-wrist-lang-proprio_2026-09-09_13-56-20"


def test_artifact_name_leaves_clean_id_unchanged_besides_prefix():
    run_id = "libero_10_dropout_2026-08-28_15-45-43"
    assert artifact_name(run_id) == f"eval-{run_id}"


def test_artifact_name_matches_wandb_charset():
    run_id = "libero_10_static+wrist+lang+proprio_2026-09-09_13-56-20"
    assert re.fullmatch(r"[A-Za-z0-9._-]+", artifact_name(run_id))
