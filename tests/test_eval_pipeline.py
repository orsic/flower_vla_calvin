"""Tests for scripts/eval_pipeline.py -- deciding which LIBERO/LIBERO-Plus evaluations a
training run needs, and resuming a partially-completed sweep.

Pure logic / filesystem tests, no MuJoCo, no model, no wandb network calls.
"""
import csv
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import eval_pipeline  # noqa: E402
import severity_sr  # noqa: E402
from eval_pipeline import (  # noqa: E402
    artifact_name,
    full_modality_combo,
    modality_combos,
    parse_overrides,
    plan_lines,
    resolve_reeval_variants,
    rotate_reeval_csvs,
    token_combos,
    wandb_run_id,
    write_severity_csv,
)

from flower.evaluation.eval_records import read_csv, write_csv  # noqa: E402


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
  project: multimodal_florence
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


# ---------------------------------------------------------------------------
# resolve_reeval_variants
# ---------------------------------------------------------------------------


def test_resolve_reeval_variants_none_means_both():
    assert resolve_reeval_variants(None, "libero_10") == {"orig", "plus"}


def test_resolve_reeval_variants_single_suite():
    assert resolve_reeval_variants("plus_libero_10", "libero_10") == {"plus"}


def test_resolve_reeval_variants_both_suites_with_whitespace():
    assert resolve_reeval_variants(" plus_libero_10 , orig_libero_10 ", "libero_10") == {"orig", "plus"}


def test_resolve_reeval_variants_duplicate_token_collapses():
    assert resolve_reeval_variants("plus_libero_10,plus_libero_10", "libero_10") == {"plus"}


@pytest.mark.parametrize("suites_arg", ["", ",", " , "])
def test_resolve_reeval_variants_empty_token_list_raises(suites_arg):
    with pytest.raises(ValueError):
        resolve_reeval_variants(suites_arg, "libero_10")


@pytest.mark.parametrize("suites_arg", ["orig", "libero_10", "orig_libero_spatial", "garbage"])
def test_resolve_reeval_variants_unrecognized_token_raises(suites_arg):
    with pytest.raises(ValueError):
        resolve_reeval_variants(suites_arg, "libero_10")


# ---------------------------------------------------------------------------
# rotate_reeval_csvs
# ---------------------------------------------------------------------------


def _seed_result_csv(train_folder, variant, benchmark="libero_10", checkpoint_name_="last"):
    path = train_folder / "eval_logs" / checkpoint_name_ / f"{variant}_{benchmark}" / "result.csv"
    write_csv(path, [_row(True, True, True, True)])
    return path


def test_rotate_reeval_csvs_only_rotates_selected_variant(tmp_path):
    train_folder = tmp_path / "run"
    orig_csv = _seed_result_csv(train_folder, "orig")
    plus_csv = _seed_result_csv(train_folder, "plus")
    checkpoint = str(train_folder / "seed_42" / "saved_models" / "last.ckpt")

    rotate_reeval_csvs({"plus"}, {}, str(train_folder), checkpoint, "libero_10")

    assert orig_csv.exists()
    assert not plus_csv.exists()
    assert len(list(plus_csv.parent.glob("results_*.csv"))) == 1


def test_rotate_reeval_csvs_csv_dir_override_both_selected_rotates_once(tmp_path):
    train_folder = tmp_path / "run"
    csv_dir = tmp_path / "shared"
    shared_csv = csv_dir / "result.csv"
    write_csv(shared_csv, [_row(True, True, True, True)])
    checkpoint = str(train_folder / "seed_42" / "saved_models" / "last.ckpt")

    backups = rotate_reeval_csvs(
        {"orig", "plus"}, {"csv_dir": str(csv_dir)}, str(train_folder), checkpoint, "libero_10"
    )

    assert len(backups) == 1
    assert not shared_csv.exists()
    assert len(list(csv_dir.glob("results_*.csv"))) == 1


def test_rotate_reeval_csvs_csv_dir_override_one_variant_raises(tmp_path):
    train_folder = tmp_path / "run"
    csv_dir = tmp_path / "shared"
    write_csv(csv_dir / "result.csv", [_row(True, True, True, True)])
    checkpoint = str(train_folder / "seed_42" / "saved_models" / "last.ckpt")

    with pytest.raises(ValueError):
        rotate_reeval_csvs({"plus"}, {"csv_dir": str(csv_dir)}, str(train_folder), checkpoint, "libero_10")


def test_rotate_reeval_csvs_missing_file_returns_none(tmp_path):
    train_folder = tmp_path / "run"
    checkpoint = str(train_folder / "seed_42" / "saved_models" / "last.ckpt")

    backups = rotate_reeval_csvs({"plus"}, {}, str(train_folder), checkpoint, "libero_10")

    assert backups == [None]


# ---------------------------------------------------------------------------
# plan_lines(reeval_variants=...) -- pure filtering, no rotation
# ---------------------------------------------------------------------------


def test_plan_lines_reeval_variants_restricts_to_plus_only(tmp_path):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=True, use_proprio=True)

    lines = plan_lines(str(train_folder), resume=False, extra_overrides=[], reeval_variants={"plus"})

    assert [svc for svc, _ in lines] == ["eval-plus"]


def test_plan_lines_reeval_variants_restricts_to_orig_only(tmp_path):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=True, use_proprio=True)

    lines = plan_lines(str(train_folder), resume=False, extra_overrides=[], reeval_variants={"orig"})

    services = [svc for svc, _ in lines]
    assert services.count("eval") == 14
    assert "eval-plus" not in services


def test_plan_lines_reeval_variants_none_is_unrestricted(tmp_path):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=False, use_proprio=False)

    lines = plan_lines(str(train_folder), resume=False, extra_overrides=[], reeval_variants=None)

    assert [svc for svc, _ in lines] == ["eval", "eval-plus"]


# ---------------------------------------------------------------------------
# main() "plan" -- --reeval / --reeval-suites CLI wiring, including the
# rotate-before-plan ordering invariant that keeps --resume from producing an
# empty plan (see eval_pipeline.py's module docstring).
# ---------------------------------------------------------------------------


def _run_plan_cli(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["eval_pipeline.py", "plan"] + args)
    eval_pipeline.main()


def test_main_reeval_with_resume_replans_rotated_suite_in_full(tmp_path, monkeypatch, capsys):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=False, use_proprio=False)
    csv_path = train_folder / "eval_logs" / "last" / "plus_libero_10" / "result.csv"
    # If rotation ran after (or never), --resume would see this "done" combo and the
    # plan would come back empty -- the exact data-loss failure mode this guards.
    write_csv(csv_path, [_row(True, True, True, True)])

    _run_plan_cli(
        monkeypatch,
        ["--train-folder", str(train_folder), "--resume", "--reeval", "--reeval-suites", "plus_libero_10"],
    )

    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line]
    assert len(lines) == 1
    assert lines[0].split("\t")[0] == "eval-plus"
    assert not csv_path.exists()
    assert len(list(csv_path.parent.glob("results_*.csv"))) == 1
    assert "reeval: rotated" in captured.err


def test_main_reeval_stdout_has_only_plan_lines(tmp_path, monkeypatch, capsys):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=False, use_proprio=False)

    _run_plan_cli(monkeypatch, ["--train-folder", str(train_folder), "--reeval"])

    captured = capsys.readouterr()
    for line in captured.out.splitlines():
        if line:
            assert "\t" in line  # every stdout line is a <service>\t<overrides...> plan line
    assert "reeval:" in captured.err


def test_main_reeval_suites_without_reeval_flag_errors(tmp_path, monkeypatch):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=False, use_proprio=False)

    with pytest.raises(SystemExit):
        _run_plan_cli(monkeypatch, ["--train-folder", str(train_folder), "--reeval-suites", "plus_libero_10"])


def test_main_reeval_invalid_suite_token_exits_with_message(tmp_path, monkeypatch):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=False, use_proprio=False)

    with pytest.raises(SystemExit) as exc_info:
        _run_plan_cli(
            monkeypatch,
            ["--train-folder", str(train_folder), "--reeval", "--reeval-suites", "plus_libero_spatial"],
        )
    assert "libero_spatial" in str(exc_info.value)


# ---------------------------------------------------------------------------
# write_severity_csv
# ---------------------------------------------------------------------------


def test_write_severity_csv_writes_next_to_plus_csv(tmp_path):
    plus_csv = tmp_path / "eval_logs" / "last" / "plus_libero_10" / "result.csv"
    write_csv(plus_csv, [_row(True, True, True, True), _row(True, True, True, True)])

    result = write_severity_csv(plus_csv)

    assert result == plus_csv.parent / "severity_sr.csv"
    assert result.exists()
    with open(result, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows  # at least the difficulty_level bin(s) for the (unclassified) category


def test_write_severity_csv_returns_none_when_plus_csv_absent(tmp_path):
    plus_csv = tmp_path / "eval_logs" / "last" / "plus_libero_10" / "result.csv"
    assert write_severity_csv(plus_csv) is None


def test_write_severity_csv_returns_none_and_warns_on_failure(tmp_path, monkeypatch, capsys):
    plus_csv = tmp_path / "eval_logs" / "last" / "plus_libero_10" / "result.csv"
    write_csv(plus_csv, [_row(True, True, True, True)])

    def _raise(*args, **kwargs):
        raise RuntimeError("boom -- LIBERO-Plus assets not downloaded")

    monkeypatch.setattr(severity_sr, "collect", _raise)

    result = write_severity_csv(plus_csv)

    assert result is None
    assert not (plus_csv.parent / "severity_sr.csv").exists()
    assert "boom" in capsys.readouterr().err


def test_write_severity_csv_with_orig_csv_adds_baseline_columns(tmp_path):
    plus_csv = tmp_path / "eval_logs" / "last" / "plus_libero_10" / "result.csv"
    plus_row = _row(True, True, True, True)
    plus_row.update({"task_name": "foo_initstate_50", "task_category": "Robot Initial States", "difficulty_level": "1"})
    write_csv(plus_csv, [plus_row])

    orig_csv = tmp_path / "eval_logs" / "last" / "orig_libero_10" / "result.csv"
    orig_row = _row(True, True, True, True)
    orig_row.update({"task_name": "foo", "init_state_idx": 0})
    write_csv(orig_csv, [orig_row])

    result = write_severity_csv(plus_csv, orig_csv)

    with open(result, newline="") as f:
        rows = list(csv.DictReader(f))
    total = next(r for r in rows if r["axis"] == "total")
    assert total["orig_n"] == "1"


def test_write_severity_csv_without_orig_csv_leaves_baseline_columns_empty(tmp_path):
    plus_csv = tmp_path / "eval_logs" / "last" / "plus_libero_10" / "result.csv"
    write_csv(plus_csv, [_row(True, True, True, True)])

    result = write_severity_csv(plus_csv)  # no orig_csv -- default None

    with open(result, newline="") as f:
        rows = list(csv.DictReader(f))
    total = next(r for r in rows if r["axis"] == "total")
    assert total["orig_n"] == ""


def test_write_severity_csv_with_absent_orig_csv_path_leaves_baseline_columns_empty(tmp_path):
    plus_csv = tmp_path / "eval_logs" / "last" / "plus_libero_10" / "result.csv"
    write_csv(plus_csv, [_row(True, True, True, True)])
    orig_csv = tmp_path / "eval_logs" / "last" / "orig_libero_10" / "result.csv"  # never written

    result = write_severity_csv(plus_csv, orig_csv)

    with open(result, newline="") as f:
        rows = list(csv.DictReader(f))
    total = next(r for r in rows if r["axis"] == "total")
    assert total["orig_n"] == ""
