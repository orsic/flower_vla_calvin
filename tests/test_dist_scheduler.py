"""Tests for scripts/dist_scheduler.py: the distributed-evaluation task queue kept on the
master machine (eligibility from W&B runs, claims, leases, retries, status)."""

import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import dist_scheduler as ds  # noqa: E402

T0 = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
SINCE = "2026-11-05"


class FakeRun:
    def __init__(
        self,
        run_id,
        state="finished",
        created_at="2026-11-06T08:00:00Z",
        host="hostA",
        epoch=None,
        args=(),
        artifacts=(),
    ):
        group, ts = run_id[: -len("_2026-11-06_08-00-00")], run_id[-len("2026-11-06_08-00-00") :]
        self.id = run_id
        self.state = state
        self.created_at = created_at
        self.metadata = {"host": host, "args": [f"hydra.run.dir=/saves/train_logs/{group}/{ts}", *args]}
        if epoch is None:
            epoch = {"libero": 39, "calvin": 34, "mimicgen": 34}[ds.benchmark_of(run_id)]
        self.summary = {"epoch": epoch}
        self._artifacts = artifacts

    def logged_artifacts(self):
        return [types.SimpleNamespace(type=t) for t in self._artifacts]


LIB = "libero_10_dropout_2026-11-06_08-00-00"
CAL = "calvin_abcd_d_dropout_2026-11-06_09-00-00"
MG = "mimicgen_dropout_2026-11-06_10-00-00"


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "run_id, bench", [(LIB, "libero"), (CAL, "calvin"), (MG, "mimicgen"), ("other_2026-11-06_08-00-00", None)]
)
def test_benchmark_of(run_id, bench):
    assert ds.benchmark_of(run_id) == bench


def test_task_from_a_finished_run():
    task = ds.task_from_run(FakeRun(LIB, host="hostB"), SINCE, [])
    assert task["run_id"] == LIB
    assert task["benchmark"] == "libero"
    assert task["host"] == "hostB"
    assert task["run_dir"] == "libero_10_dropout/2026-11-06_08-00-00"
    assert task["state"] == "todo"
    assert task["opts"] == ""


@pytest.mark.parametrize(
    "kwargs",
    [
        {"state": "running"},  # still training
        {"state": "crashed"},
        {"epoch": 0},  # finished but died at startup
        {"epoch": 38},  # one epoch short of max_epochs=40
        {"artifacts": ("evaluation",)},  # already evaluated
    ],
)
def test_ineligible_runs(kwargs):
    assert ds.task_from_run(FakeRun(LIB, **kwargs), SINCE, []) is None


def test_max_epochs_comes_from_the_config_and_the_run_args():
    assert ds.max_epochs("libero", []) == 40
    assert ds.max_epochs("calvin", []) == 35
    assert ds.max_epochs("mimicgen", []) == 35
    assert ds.max_epochs("calvin", ["seed=5", "max_epochs=10"]) == 10
    # a shortened run is complete at its own last epoch
    assert ds.task_from_run(FakeRun(CAL, epoch=9, args=("max_epochs=10",)), SINCE, []) is not None


def test_eval_opts_first_matching_glob_wins():
    rules = ds.parse_opts("libero_10_dropout_*:PIPELINE_SKIP_MULTI_OFF=1 A=2;libero_*:B=3; ;calvin_*:")
    assert ds.opts_for(LIB, rules) == "PIPELINE_SKIP_MULTI_OFF=1 A=2"
    assert ds.opts_for("libero_10_2026-11-06_08-00-00", rules) == "B=3"
    assert ds.opts_for(CAL, rules) == ""
    assert ds.opts_for(MG, rules) == ""
    assert ds.task_from_run(FakeRun(LIB), SINCE, rules)["opts"] == "PIPELINE_SKIP_MULTI_OFF=1 A=2"


def test_parse_opts_rejects_malformed_values():
    with pytest.raises(ValueError):
        ds.parse_opts("libero_*:NOT_AN_ASSIGNMENT")


# ---------------------------------------------------------------------------
# Queue operations
# ---------------------------------------------------------------------------


def _state(*runs):
    state = ds.empty_state()
    ds.refresh(state, runs, SINCE, [], T0)
    return state


def test_refresh_adds_new_runs_once_and_marks_externally_evaluated_ones_done():
    state = _state(FakeRun(LIB), FakeRun(CAL))
    assert sorted(state["tasks"]) == sorted([LIB, CAL])
    ds.refresh(state, [FakeRun(LIB), FakeRun(CAL, artifacts=("evaluation",))], SINCE, [], T0)
    assert state["tasks"][LIB]["state"] == "todo"
    assert state["tasks"][CAL]["state"] == "done"


def test_claims_go_oldest_first_and_never_twice():
    state = _state(
        FakeRun(MG, created_at="2026-11-06T10:00:00Z"),
        FakeRun(LIB, created_at="2026-11-06T08:00:00Z"),
        FakeRun(CAL, created_at="2026-11-06T09:00:00Z"),
    )
    got = [ds.claim(state, f"w:{i}", T0)["run_id"] for i in range(3)]
    assert got == [LIB, CAL, MG]
    assert ds.claim(state, "w:3", T0) is None
    assert state["tasks"][LIB]["state"] == "running" and state["tasks"][LIB]["worker"] == "w:0"
    assert state["tasks"][LIB]["attempts"] == 1


def test_expired_lease_is_requeued_and_the_stale_worker_is_refused():
    state = _state(FakeRun(LIB))
    ds.claim(state, "old:0", T0)
    assert ds.heartbeat(state, LIB, "old:0", T0 + timedelta(minutes=20))
    ds.expire_leases(state, T0 + timedelta(minutes=45), lease_min=30)
    assert state["tasks"][LIB]["state"] == "running"  # heartbeat at +20 keeps it alive
    ds.expire_leases(state, T0 + timedelta(minutes=51), lease_min=30)
    assert state["tasks"][LIB]["state"] == "todo"

    ds.claim(state, "new:1", T0 + timedelta(minutes=52))
    assert not ds.heartbeat(state, LIB, "old:0", T0 + timedelta(minutes=53))
    assert not ds.finish(state, LIB, "old:0", ok=True, msg="", now=T0 + timedelta(minutes=54))
    assert state["tasks"][LIB]["worker"] == "new:1"


def test_a_failed_task_is_retried_once_then_marked_failed():
    state = _state(FakeRun(LIB))
    ds.claim(state, "w:0", T0)
    assert ds.finish(state, LIB, "w:0", ok=False, msg="exit 1", now=T0)
    assert state["tasks"][LIB]["state"] == "todo"
    ds.claim(state, "w:1", T0)
    assert ds.finish(state, LIB, "w:1", ok=False, msg="exit 2", now=T0)
    assert state["tasks"][LIB]["state"] == "failed"
    assert state["tasks"][LIB]["error"] == "exit 2"
    assert ds.claim(state, "w:2", T0) is None


def test_done_requeue_and_skip():
    state = _state(FakeRun(LIB), FakeRun(CAL))
    ds.claim(state, "w:0", T0)
    assert ds.finish(state, LIB, "w:0", ok=True, msg="", now=T0)
    assert state["tasks"][LIB]["state"] == "done"
    ds.skip(state, CAL)
    assert state["tasks"][CAL]["state"] == "skipped"
    assert ds.claim(state, "w:1", T0) is None
    ds.requeue(state, CAL)
    assert state["tasks"][CAL]["state"] == "todo" and state["tasks"][CAL]["attempts"] == 0
    with pytest.raises(KeyError):
        ds.requeue(state, "nope")


def test_task_line_is_tab_separated_for_the_worker():
    state = ds.empty_state()
    ds.refresh(state, [FakeRun(LIB, host="hostB")], SINCE, ds.parse_opts("libero_*:X=1 Y=2"), T0)
    line = ds.task_line(ds.claim(state, "w:0", T0))
    assert line.split("\t") == [LIB, "libero", "hostB", "libero_10_dropout/2026-11-06_08-00-00", "X=1 Y=2"]


def test_status_lists_counts_and_every_task():
    state = _state(FakeRun(LIB), FakeRun(CAL), FakeRun(MG))
    ds.claim(state, "hostC:1", T0)
    out = ds.status_table(state, T0 + timedelta(minutes=7))
    assert "todo 2" in out and "running 1" in out
    for run_id in (LIB, CAL, MG):
        assert run_id in out
    assert "hostC:1" in out


def test_state_file_round_trip_under_lock(tmp_path):
    path = tmp_path / "state.json"
    with ds.locked_state(path) as state:
        ds.refresh(state, [FakeRun(LIB)], SINCE, [], T0)
    with ds.locked_state(path) as state:
        assert state["tasks"][LIB]["state"] == "todo"
        ds.claim(state, "w:0", T0)
    with ds.locked_state(path) as state:
        assert state["tasks"][LIB]["state"] == "running"


def test_since_compares_with_the_local_date_in_the_run_name_not_utc_created_at():
    """A run started at 00:00 local time on DIST_SINCE is created the previous day in
    W&B's UTC created_at; the date in its name is what DIST_SINCE refers to."""
    run = FakeRun("calvin_abcd_d_dropout_2026-11-06_00-00-18", created_at="2026-11-05T23:00:44Z")
    assert ds.task_from_run(run, "2026-11-06", []) is not None
    late = FakeRun("calvin_abcd_d_dropout_2026-11-05_23-59-59", created_at="2026-11-05T22:59:59Z")
    assert ds.task_from_run(late, "2026-11-06", []) is None
    assert ds.run_date(FakeRun(LIB)) == "2026-11-06_08-00-00"


def test_release_returns_the_task_without_using_an_attempt():
    state = _state(FakeRun(LIB))
    ds.claim(state, "w:0", T0)
    assert not ds.release(state, LIB, "other:1")  # only the owner can release it
    assert ds.release(state, LIB, "w:0")
    task = state["tasks"][LIB]
    assert task["state"] == "todo" and task["worker"] is None and task["attempts"] == 0
    assert "released by w:0" in task["error"]
    assert ds.claim(state, "w:1", T0)["run_id"] == LIB
