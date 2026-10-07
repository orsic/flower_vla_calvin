"""Tests for scripts/pro_sr.py -- the equated-subset measurement for LIBERO-PRO.

The point of these tests is that a PRO suite's number is only read against the *same*
episodes run unperturbed: orig rows outside the PRO side's key set must never enter the
baseline, and the pairing granularity must follow which init-state arm produced the rows.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import pro_sr  # noqa: E402
from pro_sr import (  # noqa: E402
    EPISODE_PAIRING,
    TASK_PAIRING,
    collect,
    combo_label,
    equated_subset,
    perturbation_vector,
)


def _row(task, episode, success, variant="pro", suite="libero_10_lan", proprio=1,
         static=1, wrist=1, lang=1):
    return {
        "libero_variant": variant,
        "suite": suite,
        "task_name": task,
        "episode_idx": str(episode),
        "success": str(success),
        "use_rgb_static": str(static),
        "use_rgb_gripper": str(wrist),
        "use_language": str(lang),
        "use_proprio": str(proprio),
    }


# ---------------------------------------------------------------------------
# perturbation_vector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "suite,vector",
    [
        ("libero_10_swap", "1,0,0,0,0"),
        ("libero_10_object", "0,1,0,0,0"),
        ("libero_10_lan", "0,0,1,0,0"),
        ("libero_10_task", "0,0,0,1,0"),
    ],
)
def test_perturbation_vector_is_the_upstream_flag_order(suite, vector):
    """(swap, object, lan, task, env) -- the same tuple LIBERO-PRO's own
    evaluation_config.yaml writes into a generated suite's log.txt."""
    assert perturbation_vector(suite) == vector


def test_perturbation_vector_of_an_unknown_suite_is_all_zero():
    assert perturbation_vector("libero_10") == "0,0,0,0,0"


# ---------------------------------------------------------------------------
# equated_subset
# ---------------------------------------------------------------------------


def test_orig_rows_outside_the_pro_key_set_are_excluded():
    """A PRO run covering 2 tasks must not be read against a 3-task orig baseline."""
    pro = [_row("task_a", 0, 1), _row("task_b", 0, 0)]
    orig = [_row("task_a", 0, 1), _row("task_b", 0, 1), _row("task_c", 0, 1)]

    pairs = equated_subset(pro, orig, EPISODE_PAIRING)

    assert sorted(pairs) == [(0, 1), (1, 1)]


def test_episode_pairing_keys_on_episode_idx():
    """The matched arm shares an init state per (task, episode), so episode 0 pairs with
    episode 0 and never with episode 1."""
    pro = [_row("task_a", 0, 1), _row("task_a", 1, 0)]
    orig = [_row("task_a", 0, 0), _row("task_a", 1, 1)]

    assert sorted(equated_subset(pro, orig, EPISODE_PAIRING)) == [(0, 1), (1, 0)]


def test_task_pairing_collapses_episodes_by_majority():
    """The native arm has no episode-level counterpart, so each side votes."""
    pro = [_row("task_a", 0, 1), _row("task_a", 1, 1), _row("task_a", 2, 0)]
    orig = [_row("task_a", 0, 0), _row("task_a", 1, 0), _row("task_a", 2, 1)]

    assert equated_subset(pro, orig, TASK_PAIRING) == [(1, 0)]


def test_task_pairing_drops_an_exact_tie():
    pro = [_row("task_a", 0, 1), _row("task_a", 1, 0)]
    orig = [_row("task_a", 0, 1), _row("task_a", 1, 1)]

    assert equated_subset(pro, orig, TASK_PAIRING) == []


def test_pairing_matches_on_modality_combo():
    """A dropout run's orig result.csv holds all 14 combos; a no-proprio PRO row must not
    pair with a full-modality orig row."""
    pro = [_row("task_a", 0, 1, proprio=0)]
    orig = [_row("task_a", 0, 1, proprio=1)]

    assert equated_subset(pro, orig, EPISODE_PAIRING) == []


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------


def test_collect_chooses_pairing_from_the_arm():
    pro_rows = (
        [_row("task_a", e, 1, variant="pro_matched") for e in range(2)]
        + [_row("task_a", e, 1, variant="pro", suite="libero_10_swap") for e in range(2)]
    )
    orig_rows = [_row("task_a", e, 0) for e in range(2)]

    by_variant = {r["libero_variant"]: r for r in collect(pro_rows, orig_rows)}

    assert by_variant["pro_matched"]["pairing"] == EPISODE_PAIRING
    assert by_variant["pro_matched"]["mcnemar_n"] == 2  # one pair per episode
    assert by_variant["pro"]["pairing"] == TASK_PAIRING
    assert by_variant["pro"]["mcnemar_n"] == 1  # one pair per task


def test_collect_without_orig_leaves_baseline_columns_empty():
    records = collect([_row("task_a", 0, 1)], None)

    assert records[0]["success_rate"] == 1.0
    assert records[0]["orig_n"] == ""
    assert records[0]["mcnemar_p"] == ""


def test_collect_reports_the_perturbation_and_task_count():
    pro_rows = [_row(t, 0, 1, suite="libero_10_task") for t in ("task_a", "task_b")]

    record = collect(pro_rows, None)[0]

    assert record["suite"] == "libero_10_task"
    assert record["perturbation"] == "task"
    assert record["perturbation_vector"] == "0,0,0,1,0"
    assert record["tasks"] == 2
    assert record["n"] == 2


def test_collect_delta_is_pro_minus_equated_orig():
    pro_rows = [_row("task_a", e, 1, variant="pro_matched") for e in range(4)]
    orig_rows = [_row("task_a", e, e % 2, variant="orig") for e in range(4)]

    record = collect(pro_rows, orig_rows)[0]

    assert record["success_rate"] == 1.0
    assert record["orig_success_rate"] == 0.5
    assert record["delta"] == pytest.approx(0.5)
    assert (record["mcnemar_b"], record["mcnemar_c"]) == (2, 0)


def test_delta_compares_both_sides_at_the_pairing_granularity():
    """Under task pairing each side collapses to one majority outcome per task over a
    task set that drops ties, so delta must be read off the pairs -- not the group's raw
    per-episode success_rate, which counts episodes the pairing excluded."""
    # task_a: PRO 2/3 (majority 1), orig 0/3 (majority 0) -> one discordant pair.
    # task_b: PRO ties 1-1 and is dropped, but its episodes still count in success_rate.
    pro_rows = (
        [_row("task_a", 0, 1), _row("task_a", 1, 1), _row("task_a", 2, 0)]
        + [_row("task_b", 0, 1), _row("task_b", 1, 0)]
    )
    orig_rows = (
        [_row("task_a", e, 0) for e in range(3)] + [_row("task_b", e, 0) for e in range(2)]
    )

    record = collect(pro_rows, orig_rows)[0]

    assert record["pairing"] == TASK_PAIRING
    assert record["success_rate"] == pytest.approx(3 / 5)  # raw episode rate, all 5 rows
    assert record["mcnemar_n"] == 1  # task_b's tie dropped
    assert record["paired_success_rate"] == 1.0
    assert record["orig_success_rate"] == 0.0
    assert record["delta"] == pytest.approx(1.0)


def test_delta_equals_b_minus_c_over_n():
    pro_rows = [_row("task_a", e, e % 2, variant="pro_matched") for e in range(4)]
    orig_rows = [_row("task_a", e, 1 if e < 1 else 0, variant="orig") for e in range(4)]

    record = collect(pro_rows, orig_rows)[0]

    assert record["delta"] == pytest.approx(
        (record["mcnemar_b"] - record["mcnemar_c"]) / record["mcnemar_n"]
    )


def test_write_csv_round_trips(tmp_path):
    out = tmp_path / "pro_sr.csv"
    pro_sr.write_csv(out, collect([_row("task_a", 0, 1)], None))

    rows = pro_sr.load_rows(out)
    assert len(rows) == 1
    assert rows[0]["suite"] == "libero_10_lan"
    assert rows[0]["perturbation_vector"] == "0,0,1,0,0"


# ---------------------------------------------------------------------------
# Matched-arm modality sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "modality_combo,label",
    [
        ((1, 1, 1, 1), "SWLP"),
        ((0, 1, 1, 1), "-WLP"),
        ((1, 1, 0, 1), "SW-P"),
        ((1, 1, 1, 0), "SWL-"),
        ((0, 0, 1, 0), "--L-"),
    ],
)
def test_combo_label(modality_combo, label):
    assert combo_label(modality_combo) == label


def test_multi_combo_file_yields_one_record_per_combo():
    """A swept matched suite merges every combo into one result.csv (the modality flags
    are KEY_COLUMNS), so collect has to split them back out."""
    pro_rows = (
        [_row("task_a", e, 1, variant="pro_matched") for e in range(2)]
        + [_row("task_a", e, 0, variant="pro_matched", lang=0) for e in range(2)]
    )

    records = collect(pro_rows, None)

    assert len(records) == 2
    assert [r["modality_combo"] for r in records] == ["1,1,1,1", "1,1,0,1"]
    assert [r["success_rate"] for r in records] == [1.0, 0.0]


def test_full_modality_sorts_first_within_a_suite():
    """eval_pipeline.modality_combos is all-on-first; the report must match, or a plain
    tuple sort leads with (0, 0, 1, 0)."""
    pro_rows = (
        [_row("task_a", 0, 1, variant="pro_matched", static=0, wrist=0, proprio=0)]
        + [_row("task_a", 1, 1, variant="pro_matched")]
        + [_row("task_a", 2, 1, variant="pro_matched", lang=0)]
    )

    assert [r["modality_combo"] for r in collect(pro_rows, None)] == [
        "1,1,1,1", "1,1,0,1", "0,0,1,0",
    ]


def test_each_combo_pairs_only_against_its_own_orig_combo():
    """The whole point of sweeping: a language-withheld PRO record must be read against
    the language-withheld orig episodes, not the full-modality ones."""
    pro_rows = [_row("task_a", e, 1, variant="pro_matched", lang=0) for e in range(4)]
    orig_rows = (
        [_row("task_a", e, 1, variant="orig") for e in range(4)]            # SWLP, all succeed
        + [_row("task_a", e, 0, variant="orig", lang=0) for e in range(4)]  # SW-P, all fail
    )

    record = collect(pro_rows, orig_rows)[0]

    assert record["modality_combo"] == "1,1,0,1"
    assert record["orig_success_rate"] == 0.0  # the SW-P rows, not the SWLP ones
    assert record["mcnemar_n"] == 4
    assert record["delta"] == pytest.approx(1.0)


def test_report_full_modality_only_filters(capsys):
    records = collect(
        [_row("task_a", 0, 1, variant="pro_matched")]
        + [_row("task_a", 1, 1, variant="pro_matched", lang=0)],
        None,
    )

    pro_sr.report(records, full_modality_only=True)
    filtered = capsys.readouterr().out
    pro_sr.report(records)
    everything = capsys.readouterr().out

    assert filtered.count("libero_10_lan") == 1
    assert everything.count("libero_10_lan") == 2
    assert "SW-P" in everything and "SW-P" not in filtered


def test_no_equated_baseline_reads_as_absent_not_zero():
    """A withheld-modality record on a non-dropout run has no orig episode to pair with
    (its orig_libero_10 only holds full modality). That is an absent baseline, not a zero
    one -- printing nan would read as a failed computation."""
    pro_rows = [_row("task_a", 0, 1, variant="pro_matched", lang=0)]
    orig_rows = [_row("task_a", 0, 1, variant="orig")]  # full modality only

    record = collect(pro_rows, orig_rows)[0]

    assert record["orig_n"] == ""
    assert record["orig_success_rate"] == ""
    assert record["delta"] == ""
    assert record["mcnemar_p"] == ""
    assert record["success_rate"] == 1.0  # the PRO side is still reported


def test_full_modality_only_is_relative_to_what_the_model_receives():
    """A use_proprio=False checkpoint records use_proprio=0 on every row, so its
    full-modality combo is (1,1,1,0). Keying on a literal (1,1,1,1) would filter such a
    run's report down to nothing."""
    records = collect(
        [_row("task_a", 0, 1, variant="pro_matched", proprio=0)]
        + [_row("task_a", 1, 1, variant="pro_matched", proprio=0, lang=0)],
        None,
    )

    kept = [r for r in records if r["modality_combo"] == "1,1,1,0"]
    assert len(kept) == 1

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pro_sr.report(records, full_modality_only=True)
    out = buf.getvalue()

    assert "SWL-" in out
    assert "SW--" not in out
