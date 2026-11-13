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
  *"mimicgen_pipeline.py restore"*|*"proprio_mean.py"*)
    [[ -n "${FAKE_PLAN_LOG:-}" ]] && echo "$joined" >> "$FAKE_PLAN_LOG"
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
    for key in (
        "PIPELINE_REEVAL", "PIPELINE_RESUME", "PIPELINE_ALL_COMBOS", "PIPELINE_SKIP_MULTI_OFF", "PIPELINE_RESTORE", "PIPELINE_MASKED",
        "FAKE_PLAN_EMPTY",
    ):
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


STUB_RERENDER = """#!/usr/bin/env bash
joined="$*"
case "$joined" in
  *"CORE_DATASETS"*)
    echo "square_d0 threading_d1 stack_d0"
    ;;
  *"huggingface-cli download"*)
    echo "download $joined" >> "$FAKE_LOG"
    [[ "$joined" == *"core/${FAKE_FAIL_DS:-none}.hdf5"* ]] && exit 1
    exit 0
    ;;
  *"prepare_mimicgen.py"*)
    echo "prepare $joined" >> "$FAKE_LOG"
    ;;
  *)
    echo "unexpected podman-compose invocation: $joined" >&2
    exit 1
    ;;
esac
"""


def _run_rerender(tmp_path, extra_env=None):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "podman-compose"
    stub.write_text(STUB_RERENDER)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "invocations.log"
    log.write_text("")
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["FAKE_LOG"] = str(log)
    env.update(extra_env or {})
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "run.sh"), "rerender-mimicgen", "-j", "2"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    return result, log.read_text().splitlines()


def test_rerender_mimicgen_downloads_then_renders_each_dataset(tmp_path):
    result, lines = _run_rerender(tmp_path)
    assert result.returncode == 0, result.stderr
    for ds in ("square_d0", "threading_d1", "stack_d0"):
        download = next(i for i, l in enumerate(lines) if l.startswith("download") and f"core/{ds}.hdf5" in l)
        prepare = next(i for i, l in enumerate(lines) if l.startswith("prepare") and l.endswith(f"prepare_mimicgen.py {ds}"))
        assert download < prepare, ds
    assert len(lines) == 6


def test_rerender_mimicgen_skips_render_and_fails_when_download_fails(tmp_path):
    result, lines = _run_rerender(tmp_path, {"FAKE_FAIL_DS": "threading_d1"})
    assert result.returncode != 0
    assert not any(l.startswith("prepare") and "threading_d1" in l for l in lines)
    assert any(l.startswith("prepare") and l.endswith("prepare_mimicgen.py square_d0") for l in lines)


STUB_TRAIN = """#!/usr/bin/env bash
joined="$*"
case "$joined" in
  *"training_libero.py"*)
    echo "train $joined" >> "$FAKE_LOG"
    ;;
  *"mimicgen_pipeline.py plan"*)
    echo "plan $joined" >> "$FAKE_LOG"
    ;;
  *"mimicgen_pipeline.py upload"*)
    echo "upload $joined" >> "$FAKE_LOG"
    ;;
  *)
    echo "unexpected podman-compose invocation: $joined" >&2
    exit 1
    ;;
esac
"""


def _run_train(tmp_path, subcommand, extra_env=None, args=("batch_size=8",)):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "podman-compose"
    stub.write_text(STUB_TRAIN)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "invocations.log"
    log.write_text("")
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["FAKE_LOG"] = str(log)
    env.pop("SKIP_PIPELINE", None)
    env.update(extra_env or {})
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "run.sh"), subcommand, *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
    )
    return result, log.read_text().splitlines()


DROPOUT_OVERRIDES = [
    "model.modality_dropout=True",
    "model.modality_dropout_keep_fraction=0.5",
    "model.modality_dropout_alphas=[1.0,1.0,1.0]",
    "model.modality_dropout_proprio_keep_p=0.5",
]


def test_train_mimicgen_dropout_passes_dropout_overrides_and_chains_pipeline(tmp_path):
    result, lines = _run_train(tmp_path, "train-mimicgen-dropout")
    assert result.returncode == 0, result.stderr
    train = next(l for l in lines if l.startswith("train"))
    tokens = train.split()
    for override in DROPOUT_OVERRIDES:
        assert override in tokens
    assert "--config-name=config_mimicgen" in tokens
    assert tokens[-1] == "batch_size=8"  # user overrides come last, so they win
    run_dir = next(t for t in tokens if t.startswith("hydra.run.dir=")).split("=", 1)[1]
    assert run_dir.startswith("/saves/train_logs/mimicgen_dropout/")
    plan = next(l for l in lines if l.startswith("plan"))
    assert f"--train-folder {run_dir}" in plan
    assert any(l.startswith("upload") for l in lines)


def test_train_mimicgen_dropout_skip_pipeline(tmp_path):
    result, lines = _run_train(tmp_path, "train-mimicgen-dropout", {"SKIP_PIPELINE": "1"})
    assert result.returncode == 0, result.stderr
    assert [l.split()[0] for l in lines] == ["train"]


def test_train_mimicgen_unchanged_run_dir_and_no_dropout(tmp_path):
    result, lines = _run_train(tmp_path, "train-mimicgen", {"SKIP_PIPELINE": "1"})
    assert result.returncode == 0, result.stderr
    tokens = lines[0].split()
    assert not any(t.startswith("model.modality_dropout") for t in tokens)
    run_dir = next(t for t in tokens if t.startswith("hydra.run.dir=")).split("=", 1)[1]
    assert run_dir.startswith("/saves/train_logs/mimicgen/")


def test_train_mimicgen_fixed_subset_names_the_run_by_its_inputs(tmp_path):
    result, lines = _run_train(
        tmp_path,
        "train-mimicgen",
        {"SKIP_PIPELINE": "1"},
        args=("model.use_proprio=True", "model.modalities.language=False"),
    )
    assert result.returncode == 0, result.stderr
    run_dir = next(t for t in lines[0].split() if t.startswith("hydra.run.dir=")).split("=", 1)[1]
    assert run_dir.startswith("/saves/train_logs/mimicgen_static+wrist+proprio/")


def test_pipeline_mimicgen_passes_resume_all_combos_and_skip_multi_off(tmp_path):
    env = {"PIPELINE_RESUME": "1", "PIPELINE_ALL_COMBOS": "1", "PIPELINE_SKIP_MULTI_OFF": "1"}
    result, _, plan, _ = _run_pipeline_mimicgen(tmp_path, env)
    assert result.returncode == 0, result.stderr
    assert all(f in plan[0] for f in ("--resume", "--all-combos", "--skip-multi-off"))
    (tmp_path / "default").mkdir()
    _, _, default, _ = _run_pipeline_mimicgen(tmp_path / "default")
    assert not any(f in default[0] for f in ("--resume", "--all-combos", "--skip-multi-off"))


def test_pipeline_mimicgen_restores_before_planning(tmp_path):
    result, evals, plan, _ = _run_pipeline_mimicgen(tmp_path, {"PIPELINE_RESTORE": "1"})
    assert result.returncode == 0, result.stderr
    assert "mimicgen_pipeline.py restore --train-folder /fake/train_dir" in plan[0]
    assert "pipeline-artifacts-mimicgen" in plan[0]
    assert "mimicgen_pipeline.py plan" in plan[1]
    assert len(evals) == 1


def test_pipeline_mimicgen_masked_restores_then_plans_masked(tmp_path):
    result, evals, plan, _ = _run_pipeline_mimicgen(tmp_path, {"PIPELINE_MASKED": "1"})
    assert result.returncode == 0, result.stderr
    assert "mimicgen_pipeline.py restore --train-folder /fake/train_dir" in plan[0]
    assert " run --rm -T shell-mimicgen python scripts/proprio_mean.py --train-folder /fake/train_dir" in plan[1]
    assert "mimicgen_pipeline.py plan" in plan[2] and "--masked" in plan[2]
