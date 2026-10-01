"""Regression test for run.sh's `pipeline-mimicgen` case, mirroring
tests/test_run_sh_pipeline.py for the LIBERO `pipeline` case: drives the real run.sh
with a stub `podman-compose` on PATH and checks every planned line gets an invocation
(not just the first -- see that file's docstring for the stdin-drain bug this guards
against), and that PIPELINE_REEVAL is threaded through to the planner.
"""
import os
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]

STUB_PODMAN_COMPOSE = """#!/usr/bin/env bash
set -euo pipefail
joined="$*"
case "$joined" in
  *"mimicgen_pipeline.py plan"*)
    [[ -n "${FAKE_PLAN_LOG:-}" ]] && echo "$joined" >> "$FAKE_PLAN_LOG"
    if [[ -n "${FAKE_PLAN_EMPTY:-}" ]]; then
        exit 0
    fi
    printf '%s\\n' "eval-mimicgen\tx=1"
    ;;
  *"flower_eval_mimicgen.py"*)
    cat > /dev/null
    echo "$joined" >> "$FAKE_EVAL_LOG"
    ;;
  *"mimicgen_pipeline.py upload"*)
    [[ -n "${FAKE_UPLOAD_LOG:-}" ]] && echo "$joined" >> "$FAKE_UPLOAD_LOG"
    ;;
  *)
    echo "unexpected podman-compose invocation: $joined" >&2
    exit 1
    ;;
esac
"""


def _run_pipeline_mimicgen(tmp_path, extra_env=None):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "podman-compose"
    stub.write_text(STUB_PODMAN_COMPOSE)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    eval_log = tmp_path / "eval_invocations.log"
    eval_log.write_text("")
    plan_log = tmp_path / "plan_invocations.log"
    plan_log.write_text("")
    upload_log = tmp_path / "upload_invocations.log"
    upload_log.write_text("")

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["FAKE_EVAL_LOG"] = str(eval_log)
    env["FAKE_PLAN_LOG"] = str(plan_log)
    env["FAKE_UPLOAD_LOG"] = str(upload_log)
    for key in ("PIPELINE_REEVAL", "FAKE_PLAN_EMPTY"):
        env.pop(key, None)
    env.update(extra_env or {})

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "run.sh"), "pipeline-mimicgen", "/fake/train_dir"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
    )
    return (
        result,
        eval_log.read_text().splitlines(),
        plan_log.read_text().splitlines(),
        upload_log.read_text().splitlines(),
    )


def test_pipeline_mimicgen_runs_the_planned_eval(tmp_path):
    result, lines, _, upload_lines = _run_pipeline_mimicgen(tmp_path)

    assert result.returncode == 0, f"run.sh failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert len(lines) == 1, f"expected 1 eval invocation, got {len(lines)}: {lines}\nstderr={result.stderr}"
    assert "x=1" in lines[0]
    assert len(upload_lines) == 1


def test_pipeline_mimicgen_omits_reeval_flag_by_default(tmp_path):
    _, _, plan_lines, _ = _run_pipeline_mimicgen(tmp_path)

    assert "--reeval" not in plan_lines[0]


def test_pipeline_mimicgen_passes_reeval_flag_when_env_set(tmp_path):
    _, _, plan_lines, _ = _run_pipeline_mimicgen(tmp_path, extra_env={"PIPELINE_REEVAL": "1"})

    assert "--reeval" in plan_lines[0]


def test_pipeline_mimicgen_empty_plan_runs_zero_evals_and_still_uploads(tmp_path):
    result, eval_lines, _, upload_lines = _run_pipeline_mimicgen(tmp_path, extra_env={"FAKE_PLAN_EMPTY": "1"})

    assert result.returncode == 0, f"run.sh failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert eval_lines == []
    assert len(upload_lines) == 1


def test_pipeline_mimicgen_requires_train_dir_arg():
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "run.sh"), "pipeline-mimicgen"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
    )
    assert result.returncode != 0
    assert "Usage" in result.stderr


STUB_ECHO_MIMICGEN_DIR = """#!/usr/bin/env bash
echo "MIMICGEN_HDF5_DIR=$MIMICGEN_HDF5_DIR"
"""


def test_mimicgen_hdf5_dir_defaults_under_data_dir(tmp_path):
    """Without an explicit MIMICGEN_HDF5_DIR, the data must land under DATA_DIR (like
    vars.env.example's `${DATA_DIR}/mimicgen_hdf5`), not the repo checkout. run.sh is
    copied into tmp_path so the real repo's vars.env isn't sourced."""
    (tmp_path / "run.sh").write_text((REPO_ROOT / "run.sh").read_text())
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "podman-compose"
    stub.write_text(STUB_ECHO_MIMICGEN_DIR)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["DATA_DIR"] = "/ssd/data"
    env.pop("MIMICGEN_HDF5_DIR", None)

    result = subprocess.run(
        ["bash", str(tmp_path / "run.sh"), "download-mimicgen", "square_d0"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "MIMICGEN_HDF5_DIR=/ssd/data/mimicgen_hdf5" in result.stdout
