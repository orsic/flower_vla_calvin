"""Regression test for run.sh's `pipeline` case.

The per-combo eval invocation (run.sh:264-276) reads scripts/eval_pipeline.py's plan
output line by line and launches one `podman-compose run` per line. If that inner
command is allowed to attach to the loop's stdin, it swallows the rest of the plan and
only the first line ever runs -- silently, with no error (see the plan file for the
full diagnosis). This test drives the real run.sh with a stub `podman-compose` on PATH
and checks that every planned line actually gets an invocation, not just the first.
"""
import os
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]

# The stub's eval branch does `cat > /dev/null`, standing in for a real container
# attaching to (and draining) whatever stdin it inherits -- the exact behavior that
# caused the bug when the loop's plan text was reachable on fd 0.
#
# The plan branch also logs the exact invocation (to $FAKE_PLAN_LOG, when set) so tests
# can assert on which flags run.sh passed through -- and, if $FAKE_PLAN_EXIT is set,
# exits with that code instead of printing a plan, to exercise run.sh's reaction to a
# failed planner (e.g. an invalid --reeval-suites token).
STUB_PODMAN_COMPOSE = """#!/usr/bin/env bash
set -euo pipefail
joined="$*"
case "$joined" in
  *"eval_pipeline.py plan"*)
    [[ -n "${FAKE_PLAN_LOG:-}" ]] && echo "$joined" >> "$FAKE_PLAN_LOG"
    if [[ -n "${FAKE_PLAN_EXIT:-}" ]]; then
        exit "$FAKE_PLAN_EXIT"
    fi
    printf '%s\\n' "eval\tx=1" "eval\tx=2" "eval-plus\tx=3"
    ;;
  *"flower_eval_libero.py"*)
    cat > /dev/null
    echo "$joined" >> "$FAKE_EVAL_LOG"
    ;;
  *"eval_pipeline.py upload"*)
    :
    ;;
  *)
    echo "unexpected podman-compose invocation: $joined" >&2
    exit 1
    ;;
esac
"""


def _run_pipeline(tmp_path, extra_env=None):
    """Run `./run.sh pipeline /fake/train_dir` against the stub, returning
    (result, eval_invocations, plan_invocations)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "podman-compose"
    stub.write_text(STUB_PODMAN_COMPOSE)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    eval_log = tmp_path / "eval_invocations.log"
    eval_log.write_text("")
    plan_log = tmp_path / "plan_invocations.log"
    plan_log.write_text("")

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["FAKE_EVAL_LOG"] = str(eval_log)
    env["FAKE_PLAN_LOG"] = str(plan_log)
    for key in ("PIPELINE_RESUME", "PIPELINE_REEVAL", "PIPELINE_REEVAL_SUITES", "FAKE_PLAN_EXIT"):
        env.pop(key, None)
    env.update(extra_env or {})

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "run.sh"), "pipeline", "/fake/train_dir"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
    )
    return result, eval_log.read_text().splitlines(), plan_log.read_text().splitlines()


def test_pipeline_runs_every_planned_eval(tmp_path):
    """A 3-line plan (2 eval + 1 eval-plus) must produce 3 eval invocations, not 1."""
    result, lines, _ = _run_pipeline(tmp_path)

    assert result.returncode == 0, f"run.sh failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert len(lines) == 3, (
        f"expected 3 eval invocations, got {len(lines)}: {lines}\nstderr={result.stderr}"
    )
    assert any("x=1" in line for line in lines)
    assert any("x=2" in line for line in lines)
    assert any("x=3" in line for line in lines)


def test_pipeline_omits_reeval_flags_by_default(tmp_path):
    _, _, plan_lines = _run_pipeline(tmp_path)

    assert "--reeval" not in plan_lines[0]
    assert "--reeval-suites" not in plan_lines[0]


def test_pipeline_passes_reeval_flag_only_when_suites_unset(tmp_path):
    _, _, plan_lines = _run_pipeline(tmp_path, extra_env={"PIPELINE_REEVAL": "1"})

    assert "--reeval" in plan_lines[0]
    assert "--reeval-suites" not in plan_lines[0]


def test_pipeline_passes_reeval_and_suites_when_both_set(tmp_path):
    _, _, plan_lines = _run_pipeline(
        tmp_path, extra_env={"PIPELINE_REEVAL": "1", "PIPELINE_REEVAL_SUITES": "plus_libero_10"}
    )

    assert "--reeval" in plan_lines[0]
    assert "--reeval-suites plus_libero_10" in plan_lines[0]


def test_pipeline_aborts_when_plan_fails(tmp_path):
    """An invalid --reeval-suites token makes the real planner exit non-zero; run.sh must
    propagate that failure rather than proceeding with an empty plan."""
    result, eval_lines, _ = _run_pipeline(tmp_path, extra_env={"FAKE_PLAN_EXIT": "1"})

    assert result.returncode != 0
    assert eval_lines == []
