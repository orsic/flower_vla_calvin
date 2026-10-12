"""Regression tests for run.sh's CALVIN cases, mirroring tests/test_run_sh_mimicgen.py:
drives the real run.sh with a stub `podman-compose` on PATH.

- pipeline-calvin runs every planned line (not just the first) and threads PIPELINE_REEVAL.
- train-calvin / train-calvin-dropout launch training_libero.py with config_calvin (the
  dropout variant with the modality-dropout overrides before the user's own) and chain
  pipeline-calvin on the run dir unless SKIP_PIPELINE is set.
"""
import os
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]

STUB_PODMAN_COMPOSE = """#!/usr/bin/env bash
set -euo pipefail
joined="$*"
echo "$joined" >> "$FAKE_LOG"
case "$joined" in
  *"calvin_pipeline.py plan"*)
    printf '%s\\n' "eval-calvin\tcombo=1" "eval-calvin\tcombo=2" "eval-calvin\tcombo=3"
    ;;
  *"flower_evaluate.py"*)
    cat > /dev/null
    ;;
esac
"""


def _run(tmp_path, args, extra_env=None):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "podman-compose"
    stub.write_text(STUB_PODMAN_COMPOSE)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    log = tmp_path / "invocations.log"
    log.write_text("")

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["FAKE_LOG"] = str(log)
    for key in ("PIPELINE_REEVAL", "SKIP_PIPELINE"):
        env.pop(key, None)
    env.update(extra_env or {})
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "run.sh"), *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
    )
    assert result.returncode == 0, f"run.sh failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    return log.read_text().splitlines()


def test_pipeline_calvin_runs_every_planned_eval_then_uploads(tmp_path):
    lines = _run(tmp_path, ["pipeline-calvin", "/fake/train_dir"])

    assert "calvin_pipeline.py plan --train-folder /fake/train_dir" in lines[0]
    assert "--reeval" not in lines[0]
    evals = [line for line in lines if "flower_evaluate.py" in line]
    assert len(evals) == 3
    assert [f"combo={i}" in line for i, line in enumerate(evals, 1)] == [True] * 3
    assert all(" run --rm -T eval-calvin python " in line for line in evals)
    assert "calvin_pipeline.py upload --train-folder /fake/train_dir" in lines[-1]
    assert "pipeline-artifacts-calvin" in lines[-1]


def test_pipeline_calvin_passes_reeval_flag_when_env_set(tmp_path):
    lines = _run(tmp_path, ["pipeline-calvin", "/fake/train_dir"], {"PIPELINE_REEVAL": "1"})

    assert "--reeval" in lines[0]


def test_train_calvin_dropout_overrides_and_chained_pipeline(tmp_path):
    lines = _run(tmp_path, ["train-calvin-dropout", "accumulate_grad_batches=2"])

    train = lines[0]
    assert " run --rm train-calvin python flower/training_libero.py --config-name=config_calvin " in train
    assert "model.modality_dropout=True" in train
    assert train.index("model.modality_dropout=True") < train.index("accumulate_grad_batches=2")
    run_dir = next(tok for tok in train.split() if tok.startswith("hydra.run.dir="))[len("hydra.run.dir="):]
    assert run_dir.startswith("/saves/train_logs/calvin_abcd_d_dropout/")
    assert f"calvin_pipeline.py plan --train-folder {run_dir}" in lines[1]


def test_train_calvin_baseline_skip_pipeline(tmp_path):
    lines = _run(tmp_path, ["train-calvin"], {"SKIP_PIPELINE": "1"})

    assert len(lines) == 1
    assert "model.modality_dropout" not in lines[0]
    assert "hydra.run.dir=/saves/train_logs/calvin_abcd_d/" in lines[0]


def test_open_loop_calvin_runs_the_scorer_in_eval_calvin(tmp_path):
    lines = _run(tmp_path, ["open-loop-calvin", "/fake/train_dir", "--max-windows", "3"])

    assert len(lines) == 1
    assert " run --rm -T eval-calvin python scripts/open_loop_calvin.py --train-folder /fake/train_dir " in lines[0]
    assert lines[0].endswith("--max-windows 3")


def test_analyze_open_loop_runs_the_analysis(tmp_path):
    lines = _run(tmp_path, ["analyze-open-loop", "--n-bins", "5"])

    assert len(lines) == 1
    assert " run --rm -T shell-calvin python scripts/open_loop_analysis.py --n-bins 5" in lines[0]
