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
    MODALITY_OFF_VARIANTS,
    all_variants,
    artifact_members,
    artifact_name,
    full_modality_combo,
    modality_combos,
    modality_off_combo,
    modality_off_csv_paths,
    modality_off_member,
    modality_off_severity_member,
    parse_overrides,
    plan_lines,
    resolve_reeval_variants,
    rotate_reeval_csvs,
    suite_dir_name,
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
# modality-off naming helpers -- suite_dir_name / modality_off_combo / all_variants /
# modality_off_member / modality_off_severity_member
# ---------------------------------------------------------------------------


def test_suite_dir_name_orig_and_plus_unchanged():
    assert suite_dir_name("orig", "libero_10") == "orig_libero_10"
    assert suite_dir_name("plus", "libero_10") == "plus_libero_10"


@pytest.mark.parametrize(
    "variant,expected_suffix",
    [
        ("plus_no_static", "no_static"),
        ("plus_no_wrist", "no_wrist"),
        ("plus_no_lang", "no_lang"),
        ("plus_no_proprio", "no_proprio"),
    ],
)
def test_suite_dir_name_modality_off_puts_benchmark_in_the_middle(variant, expected_suffix):
    assert suite_dir_name(variant, "libero_10") == f"plus_libero_10_{expected_suffix}"


def test_suite_dir_name_follows_benchmark_override():
    assert suite_dir_name("plus_no_lang", "libero_spatial") == "plus_libero_spatial_no_lang"


@pytest.mark.parametrize("variant", list(MODALITY_OFF_VARIANTS))
def test_modality_off_combo_turns_off_exactly_one(variant):
    combo = modality_off_combo(variant)
    off_count = sum(1 for v in combo.values() if v is False)
    assert off_count == 1
    key, _suffix = MODALITY_OFF_VARIANTS[variant]
    assert combo[key] is False


def test_all_variants_is_six_with_proprio_five_without():
    with_proprio = all_variants(True)
    without_proprio = all_variants(False)
    assert len(with_proprio) == 6
    assert len(without_proprio) == 5
    assert "plus_no_proprio" in with_proprio
    assert "plus_no_proprio" not in without_proprio


def test_modality_off_member_names_are_distinct_and_prefixed():
    csv_names = {modality_off_member(v) for v in MODALITY_OFF_VARIANTS}
    sev_names = {modality_off_severity_member(v) for v in MODALITY_OFF_VARIANTS}
    assert csv_names == {
        "libero_plus_no_static.csv", "libero_plus_no_wrist.csv",
        "libero_plus_no_lang.csv", "libero_plus_no_proprio.csv",
    }
    assert sev_names == {
        "severity_sr_no_static.csv", "severity_sr_no_wrist.csv",
        "severity_sr_no_lang.csv", "severity_sr_no_proprio.csv",
    }
    assert not csv_names & sev_names
    assert "libero_plus.csv" not in csv_names
    assert "severity_sr.csv" not in sev_names


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

    # One full-modality eval-plus, then one per withheld modality -- no_proprio is
    # excluded since this checkpoint never receives proprioception.
    services = [svc for svc, _ in lines]
    assert services == ["eval"] + ["eval-plus"] * 4


def test_plan_lines_dropout_no_proprio_is_seven_plus_four(tmp_path):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=True, use_proprio=False)

    lines = plan_lines(str(train_folder), resume=False, extra_overrides=[])

    services = [svc for svc, _ in lines]
    assert services.count("eval") == 7
    assert services.count("eval-plus") == 4
    assert services[-1] == "eval-plus"


def test_plan_lines_dropout_with_proprio_is_fourteen_plus_five(tmp_path):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=True, use_proprio=True)

    lines = plan_lines(str(train_folder), resume=False, extra_overrides=[])

    services = [svc for svc, _ in lines]
    assert services.count("eval") == 14
    assert services.count("eval-plus") == 5


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

    # The one combo a non-dropout run needs on LIBERO is already in result.csv; the
    # full-modality eval-plus and all 4 modality-off eval-plus lines remain (untouched).
    assert [svc for svc, _ in lines] == ["eval-plus"] * 5


def test_plan_lines_resume_keeps_uncovered_combo(tmp_path):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=True, use_proprio=True)
    csv_path = train_folder / "eval_logs" / "last" / "orig_libero_10" / "result.csv"
    # Only the all-on combo has been evaluated; the other 13 dropout combos have not.
    write_csv(csv_path, [_row(True, True, True, True)])

    lines = plan_lines(str(train_folder), resume=True, extra_overrides=[])

    assert [svc for svc, _ in lines].count("eval") == 13
    assert [svc for svc, _ in lines].count("eval-plus") == 5


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
    assert [svc for svc, _ in lines].count("eval-plus") == 4


# ---------------------------------------------------------------------------
# plan_lines -- modality-off eval-plus lines
# ---------------------------------------------------------------------------


def _modality_off_lines(lines):
    """The 4 (or fewer) eval-plus lines that carry a csv_dir= override -- the
    full-modality eval-plus line never does (see test_plan_lines_full_plus_line_has_no_csv_dir_override)."""
    return [
        (svc, overrides)
        for svc, overrides in lines
        if svc == "eval-plus" and any(o.startswith("csv_dir=") for o in overrides)
    ]


def test_plan_lines_modality_off_lines_withhold_exactly_one_modality(tmp_path):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=False, use_proprio=True)

    lines = plan_lines(str(train_folder), resume=False, extra_overrides=[])
    modality_off_lines = _modality_off_lines(lines)

    assert len(modality_off_lines) == 4
    seen_off_keys = set()
    for _svc, overrides in modality_off_lines:
        flags = {o.split("=")[0].split(".")[1]: o.split("=")[1] for o in overrides if o.startswith("eval_modalities.")}
        off_keys = [k for k, v in flags.items() if v == "False"]
        assert len(off_keys) == 1
        seen_off_keys.add(off_keys[0])
    assert seen_off_keys == {"rgb_static", "rgb_gripper", "language", "proprio"}


def test_plan_lines_modality_off_lines_carry_their_own_csv_dir(tmp_path):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=False, use_proprio=True)

    lines = plan_lines(str(train_folder), resume=False, extra_overrides=[])
    modality_off_overrides = [overrides for _svc, overrides in _modality_off_lines(lines)]

    expected_dir = train_folder / "eval_logs" / "last" / "plus_libero_10_no_static"
    assert any(f"csv_dir={expected_dir}" in overrides for overrides in modality_off_overrides)


def test_plan_lines_full_plus_line_has_no_csv_dir_override(tmp_path):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=False, use_proprio=True)

    lines = plan_lines(str(train_folder), resume=False, extra_overrides=[])
    full_plus_lines = [
        overrides for svc, overrides in lines
        if svc == "eval-plus" and not any(o.startswith("csv_dir=") for o in overrides)
    ]

    assert len(full_plus_lines) == 1


def test_plan_lines_modality_off_lines_keep_the_real_benchmark_name(tmp_path):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=False, use_proprio=True)

    lines = plan_lines(str(train_folder), resume=False, extra_overrides=[])

    for _svc, overrides in _modality_off_lines(lines):
        assert "benchmark_name=libero_10" in overrides


def test_plan_lines_user_overrides_still_last_on_modality_off_lines(tmp_path):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=False, use_proprio=True)

    lines = plan_lines(str(train_folder), resume=False, extra_overrides=["n_eval=5", "eval_batch_size=32"])

    for _svc, overrides in _modality_off_lines(lines):
        assert overrides[-2:] == ["n_eval=5", "eval_batch_size=32"]


def test_plan_lines_user_csv_dir_override_wins_over_the_emitted_one(tmp_path):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=False, use_proprio=True)

    lines = plan_lines(str(train_folder), resume=False, extra_overrides=["csv_dir=/x"])

    # A modality-off line has both the planner's own csv_dir= and the user's -- the
    # user's must be the last one on the line (last-wins Hydra convention).
    modality_off_lines = [
        overrides for svc, overrides in lines
        if svc == "eval-plus" and overrides.count("csv_dir=/x") == 1 and any(
            o.startswith("csv_dir=") and o != "csv_dir=/x" for o in overrides
        )
    ]
    assert len(modality_off_lines) == 4
    for overrides in modality_off_lines:
        csv_dir_overrides = [o for o in overrides if o.startswith("csv_dir=")]
        assert csv_dir_overrides[-1] == "csv_dir=/x"


def test_plan_lines_skips_no_proprio_variant_when_model_has_no_proprio(tmp_path):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=False, use_proprio=False)

    lines = plan_lines(str(train_folder), resume=False, extra_overrides=[])

    for _svc, overrides in lines:
        assert "eval_modalities.proprio=False" not in overrides
    assert not any("no_proprio" in o for _svc, overrides in lines for o in overrides)


def test_plan_lines_skip_modality_off_flag_suppresses_all_four(tmp_path):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=False, use_proprio=True)

    lines = plan_lines(str(train_folder), resume=False, extra_overrides=[], skip_modality_off=True)

    assert [svc for svc, _ in lines] == ["eval", "eval-plus"]


def test_plan_lines_resume_skips_a_completed_modality_off_suite(tmp_path):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=False, use_proprio=True)
    csv_path = train_folder / "eval_logs" / "last" / "plus_libero_10_no_lang" / "result.csv"
    write_csv(csv_path, [_row(True, True, False, True)])

    lines = plan_lines(str(train_folder), resume=True, extra_overrides=[])

    # orig ("eval") + full-plus stay (not seeded); no_static/no_wrist/no_proprio stay;
    # no_lang is done.
    assert len(lines) == 5
    for _svc, overrides in lines:
        assert "csv_dir=" + str(train_folder / "eval_logs" / "last" / "plus_libero_10_no_lang") not in "".join(overrides)


def test_plan_lines_resume_modality_off_matches_effective_proprio(tmp_path):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=False, use_proprio=False)
    csv_path = train_folder / "eval_logs" / "last" / "plus_libero_10_no_static" / "result.csv"
    write_csv(csv_path, [_row(False, True, True, False)])  # use_proprio=0, as a real run would write

    lines = plan_lines(str(train_folder), resume=True, extra_overrides=[])

    services_and_overrides = [(svc, overrides) for svc, overrides in lines]
    assert not any(
        any("no_static" in o for o in overrides) for _svc, overrides in services_and_overrides
    )


def test_plan_lines_resume_full_plus_csv_does_not_mark_modality_off_done(tmp_path):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=False, use_proprio=True)
    csv_path = train_folder / "eval_logs" / "last" / "plus_libero_10" / "result.csv"
    write_csv(csv_path, [_row(True, True, True, True)])

    lines = plan_lines(str(train_folder), resume=True, extra_overrides=[])

    # orig ("eval") isn't seeded, so it stays; full-plus is done; all 4 modality-off stay.
    assert [svc for svc, _ in lines] == ["eval"] + ["eval-plus"] * 4


def test_plan_lines_resume_modality_off_csv_does_not_mark_full_plus_done(tmp_path):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=False, use_proprio=True)
    csv_path = train_folder / "eval_logs" / "last" / "plus_libero_10_no_static" / "result.csv"
    write_csv(csv_path, [_row(False, True, True, True)])

    lines = plan_lines(str(train_folder), resume=True, extra_overrides=[])

    # orig ("eval") + full-plus + no_wrist + no_lang + no_proprio; no_static is done.
    assert len(lines) == 5


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


def test_resolve_reeval_variants_none_means_every_suite():
    assert resolve_reeval_variants(None, "libero_10") == {
        "orig", "plus", "plus_no_static", "plus_no_wrist", "plus_no_lang", "plus_no_proprio",
    }


def test_resolve_reeval_variants_none_excludes_no_proprio_without_proprio():
    assert resolve_reeval_variants(None, "libero_10", use_proprio=False) == {
        "orig", "plus", "plus_no_static", "plus_no_wrist", "plus_no_lang",
    }


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


@pytest.mark.parametrize(
    "suites_arg",
    ["orig", "libero_10", "orig_libero_spatial", "garbage", "plus_libero_10_no_gripper", "plus_no_static"],
)
def test_resolve_reeval_variants_unrecognized_token_raises(suites_arg):
    with pytest.raises(ValueError):
        resolve_reeval_variants(suites_arg, "libero_10")


@pytest.mark.parametrize(
    "suites_arg,expected",
    [
        ("plus_libero_10_no_static", {"plus_no_static"}),
        ("plus_libero_10_no_wrist", {"plus_no_wrist"}),
        ("plus_libero_10_no_lang", {"plus_no_lang"}),
        ("plus_libero_10_no_proprio", {"plus_no_proprio"}),
        ("plus_libero_10,plus_libero_10_no_lang", {"plus", "plus_no_lang"}),
    ],
)
def test_resolve_reeval_variants_modality_off_suite_names(suites_arg, expected):
    assert resolve_reeval_variants(suites_arg, "libero_10") == expected


def test_resolve_reeval_variants_no_proprio_suite_is_skipped_with_a_note_on_a_no_proprio_run(capsys):
    result = resolve_reeval_variants(
        "plus_libero_10_no_proprio,plus_libero_10_no_lang", "libero_10", use_proprio=False
    )
    assert result == {"plus_no_lang"}
    assert "use_proprio=False" in capsys.readouterr().err


def test_resolve_reeval_variants_only_inapplicable_suite_raises():
    with pytest.raises(ValueError):
        resolve_reeval_variants("plus_libero_10_no_proprio", "libero_10", use_proprio=False)


def test_resolve_reeval_variants_error_message_lists_the_new_suites():
    with pytest.raises(ValueError, match="plus_libero_10_no_static"):
        resolve_reeval_variants("garbage", "libero_10")


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


def test_rotate_reeval_csvs_csv_dir_override_every_variant_selected_rotates_once(tmp_path):
    train_folder = tmp_path / "run"
    csv_dir = tmp_path / "shared"
    shared_csv = csv_dir / "result.csv"
    write_csv(shared_csv, [_row(True, True, True, True)])
    checkpoint = str(train_folder / "seed_42" / "saved_models" / "last.ckpt")

    backups = rotate_reeval_csvs(
        set(eval_pipeline.all_variants(True)),
        {"csv_dir": str(csv_dir)},
        str(train_folder),
        checkpoint,
        "libero_10",
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

    assert [svc for svc, _ in lines] == ["eval"] + ["eval-plus"] * 4


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


def test_main_reeval_modality_off_suite_replans_only_that_line(tmp_path, monkeypatch, capsys):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=False, use_proprio=True)
    csv_path = train_folder / "eval_logs" / "last" / "plus_libero_10_no_lang" / "result.csv"
    write_csv(csv_path, [_row(True, True, False, True)])

    _run_plan_cli(
        monkeypatch,
        ["--train-folder", str(train_folder), "--resume", "--reeval", "--reeval-suites", "plus_libero_10_no_lang"],
    )

    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line]
    assert len(lines) == 1
    fields = lines[0].split("\t")
    assert fields[0] == "eval-plus"
    assert "eval_modalities.language=False" in fields
    assert not csv_path.exists()
    assert len(list(csv_path.parent.glob("results_*.csv"))) == 1


def test_main_skip_modality_off_flag_suppresses_the_four_lines(tmp_path, monkeypatch, capsys):
    train_folder = tmp_path / "run"
    _write_train_cfg(train_folder, dropout=False, use_proprio=True)

    _run_plan_cli(monkeypatch, ["--train-folder", str(train_folder), "--skip-modality-off"])

    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    assert [line.split("\t")[0] for line in lines] == ["eval", "eval-plus"]


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


def test_write_severity_csv_writes_into_a_modality_off_suite_dir(tmp_path):
    modality_off_csv = tmp_path / "eval_logs" / "last" / "plus_libero_10_no_static" / "result.csv"
    write_csv(modality_off_csv, [_row(False, True, True, True)])

    result = write_severity_csv(modality_off_csv)

    assert result == modality_off_csv.parent / "severity_sr.csv"
    assert result.parent.name == "plus_libero_10_no_static"


# ---------------------------------------------------------------------------
# modality_off_csv_paths / artifact_members
# ---------------------------------------------------------------------------


def test_modality_off_csv_paths_point_at_the_new_suite_dirs(tmp_path):
    train_folder = tmp_path / "run"
    checkpoint = str(train_folder / "seed_42" / "saved_models" / "last.ckpt")

    paths = modality_off_csv_paths(str(train_folder), checkpoint, "libero_10", use_proprio=True)

    assert set(paths) == set(MODALITY_OFF_VARIANTS)
    assert paths["plus_no_static"].parent.name == "plus_libero_10_no_static"


def test_modality_off_csv_paths_omits_no_proprio_when_not_applicable(tmp_path):
    train_folder = tmp_path / "run"
    checkpoint = str(train_folder / "seed_42" / "saved_models" / "last.ckpt")

    paths = modality_off_csv_paths(str(train_folder), checkpoint, "libero_10", use_proprio=False)

    assert "plus_no_proprio" not in paths
    assert len(paths) == 3


def _seed_modality_off_csv(train_folder, suffix, benchmark="libero_10", checkpoint_name_="last", row=None):
    path = train_folder / "eval_logs" / checkpoint_name_ / f"plus_{benchmark}_{suffix}" / "result.csv"
    write_csv(path, [row or _row(True, True, True, True)])
    return path


def test_artifact_members_includes_every_present_modality_off_csv_and_its_severity_sibling(tmp_path):
    train_folder = tmp_path / "run"
    orig_csv = train_folder / "eval_logs" / "last" / "orig_libero_10" / "result.csv"
    plus_csv = train_folder / "eval_logs" / "last" / "plus_libero_10" / "result.csv"
    write_csv(plus_csv, [_row(True, True, True, True)])
    modality_off_csvs = {
        "plus_no_static": _seed_modality_off_csv(train_folder, "no_static"),
        "plus_no_wrist": _seed_modality_off_csv(train_folder, "no_wrist"),
        "plus_no_lang": _seed_modality_off_csv(train_folder, "no_lang"),
        "plus_no_proprio": _seed_modality_off_csv(train_folder, "no_proprio"),
    }

    members = artifact_members(orig_csv, plus_csv, modality_off_csvs, pid_txt=None)
    names = {name for _path, name in members}

    assert names == {
        "libero_plus.csv", "severity_sr.csv",
        "libero_plus_no_static.csv", "severity_sr_no_static.csv",
        "libero_plus_no_wrist.csv", "severity_sr_no_wrist.csv",
        "libero_plus_no_lang.csv", "severity_sr_no_lang.csv",
        "libero_plus_no_proprio.csv", "severity_sr_no_proprio.csv",
    }


def test_artifact_members_member_names_are_unique(tmp_path):
    train_folder = tmp_path / "run"
    orig_csv = train_folder / "eval_logs" / "last" / "orig_libero_10" / "result.csv"
    write_csv(orig_csv, [_row(True, True, True, True)])
    plus_csv = train_folder / "eval_logs" / "last" / "plus_libero_10" / "result.csv"
    write_csv(plus_csv, [_row(True, True, True, True)])
    modality_off_csvs = {"plus_no_static": _seed_modality_off_csv(train_folder, "no_static")}
    pid_txt = train_folder / "eval_logs" / "last" / "orig_libero_10" / "pid_modality.txt"
    pid_txt.write_text("dummy")

    members = artifact_members(orig_csv, plus_csv, modality_off_csvs, pid_txt)
    names = [name for _path, name in members]

    assert len(names) == len(set(names))


def test_artifact_members_skips_absent_modality_off_csvs(tmp_path):
    train_folder = tmp_path / "run"
    orig_csv = train_folder / "eval_logs" / "last" / "orig_libero_10" / "result.csv"
    plus_csv = train_folder / "eval_logs" / "last" / "plus_libero_10" / "result.csv"
    write_csv(plus_csv, [_row(True, True, True, True)])
    modality_off_csvs = {
        "plus_no_static": _seed_modality_off_csv(train_folder, "no_static"),
        "plus_no_wrist": train_folder / "eval_logs" / "last" / "plus_libero_10_no_wrist" / "result.csv",  # not written
    }

    members = artifact_members(orig_csv, plus_csv, modality_off_csvs, pid_txt=None)
    names = {name for _path, name in members}

    assert "libero_plus_no_static.csv" in names
    assert "libero_plus_no_wrist.csv" not in names
    assert "severity_sr_no_wrist.csv" not in names


def test_artifact_members_severity_failure_drops_only_that_member(tmp_path, monkeypatch, capsys):
    train_folder = tmp_path / "run"
    orig_csv = train_folder / "eval_logs" / "last" / "orig_libero_10" / "result.csv"
    plus_csv = train_folder / "eval_logs" / "last" / "plus_libero_10" / "result.csv"
    write_csv(plus_csv, [_row(True, True, True, True)])
    modality_off_csvs = {"plus_no_static": _seed_modality_off_csv(train_folder, "no_static")}

    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(severity_sr, "collect", _raise)

    members = artifact_members(orig_csv, plus_csv, modality_off_csvs, pid_txt=None)
    names = {name for _path, name in members}

    assert "libero_plus_no_static.csv" in names
    assert not any(name.startswith("severity_sr") for name in names)
    assert "boom" in capsys.readouterr().err


def test_artifact_members_empty_when_nothing_exists(tmp_path):
    train_folder = tmp_path / "run"
    orig_csv = train_folder / "eval_logs" / "last" / "orig_libero_10" / "result.csv"
    plus_csv = train_folder / "eval_logs" / "last" / "plus_libero_10" / "result.csv"

    assert artifact_members(orig_csv, plus_csv, {}, pid_txt=None) == []


def test_artifact_members_modality_off_severity_uses_the_orig_baseline(tmp_path):
    train_folder = tmp_path / "run"
    orig_csv = train_folder / "eval_logs" / "last" / "orig_libero_10" / "result.csv"
    orig_row = _row(False, True, True, True)
    orig_row.update({"task_name": "foo", "init_state_idx": 0})
    write_csv(orig_csv, [orig_row])
    plus_csv = train_folder / "eval_logs" / "last" / "plus_libero_10" / "result.csv"
    write_csv(plus_csv, [_row(True, True, True, True)])
    modality_off_row = _row(False, True, True, True)
    modality_off_row.update({"task_name": "foo_initstate_50", "task_category": "Robot Initial States", "difficulty_level": "1"})
    modality_off_csvs = {"plus_no_static": _seed_modality_off_csv(train_folder, "no_static", row=modality_off_row)}

    members = artifact_members(orig_csv, plus_csv, modality_off_csvs, pid_txt=None)
    sev_path = next(path for path, name in members if name == "severity_sr_no_static.csv")

    with open(sev_path, newline="") as f:
        rows = list(csv.DictReader(f))
    total = next(r for r in rows if r["axis"] == "total")
    assert total["orig_n"] == "1"
