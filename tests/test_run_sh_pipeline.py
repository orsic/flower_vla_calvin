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
STUB_PODMAN_COMPOSE = """#!/usr/bin/env bash
set -euo pipefail
joined="$*"
case "$joined" in
  *"eval_pipeline.py plan"*)
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


def test_pipeline_runs_every_planned_eval(tmp_path):
    """A 3-line plan (2 eval + 1 eval-plus) must produce 3 eval invocations, not 1."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "podman-compose"
    stub.write_text(STUB_PODMAN_COMPOSE)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    log_path = tmp_path / "eval_invocations.log"
    log_path.write_text("")

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["FAKE_EVAL_LOG"] = str(log_path)

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "run.sh"), "pipeline", "/fake/train_dir"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
    )

    assert result.returncode == 0, f"run.sh failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    lines = log_path.read_text().splitlines()
    assert len(lines) == 3, (
        f"expected 3 eval invocations, got {len(lines)}: {lines}\nstderr={result.stderr}"
    )
    assert any("x=1" in line for line in lines)
    assert any("x=2" in line for line in lines)
    assert any("x=3" in line for line in lines)
