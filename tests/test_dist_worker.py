"""Tests for scripts/dist_worker.sh: an evaluation worker claims a task from the master
over ssh, fetches the run (unless it trained on this machine), runs the benchmark's
pipeline on its GPU with the task's options, reports the outcome and deletes the copied
checkpoint. ssh, rsync and run.sh are stubs on PATH; DIST_ONCE=1 makes every GPU loop
do a single claim cycle, and DIST_SKIP_AGENT=1 skips the ssh-agent/ssh-add setup."""

import os
import signal
import stat
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]
RUN = "libero_10_dropout_2026-11-06_08-00-00"
RUN_DIR = "libero_10_dropout/2026-11-06_08-00-00"

STUB_SSH = """#!/usr/bin/env bash
echo "$*" >> "$FAKE_SSH_LOG"
case "$*" in
  *"dist-sched"*"claim"*)
    # hand out the fake task once, then nothing; mkdir is atomic, so two GPU loops
    # claiming at the same moment can't both get it (the real queue holds a lock)
    if [[ -n "${FAKE_TASK:-}" ]] && mkdir "$FAKE_SSH_LOG.claimed" 2>/dev/null; then
        printf '%b\\n' "$FAKE_TASK"
    fi ;;
esac
exit 0
"""

STUB_RSYNC = """#!/usr/bin/env bash
echo "$*" >> "$FAKE_RSYNC_LOG"
dest="${@: -1}"
mkdir -p "$dest/.hydra" "$dest/seed_42/saved_models"
echo cfg > "$dest/.hydra/config.yaml"
echo ckpt > "$dest/seed_42/saved_models/last.ckpt"
"""

STUB_RUN_SH = """#!/usr/bin/env bash
echo "CVD=${CUDA_VISIBLE_DEVICES:-} RESUME=${PIPELINE_RESUME:-} MULTI=${PIPELINE_SKIP_MULTI_OFF:-} ARGS=$*" >> "$FAKE_RUN_LOG"
exit "${FAKE_RC:-0}"
"""


def _stub(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _run_worker(
    tmp_path, task_host="hostB", fake_rc=0, task=True, opts="PIPELINE_SKIP_MULTI_OFF=1", gpus="1", bench="libero"
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    _stub(bin_dir / "ssh", STUB_SSH)
    _stub(bin_dir / "rsync", STUB_RSYNC)
    run_sh = tmp_path / "run.sh"
    _stub(run_sh, STUB_RUN_SH)
    saves = tmp_path / "saves"
    saves.mkdir(exist_ok=True)
    logs = {k: tmp_path / f"{k}.log" for k in ("ssh", "rsync", "run")}
    for p in logs.values():
        p.write_text("")

    env = dict(os.environ)
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}",
        FAKE_SSH_LOG=str(logs["ssh"]),
        FAKE_RSYNC_LOG=str(logs["rsync"]),
        FAKE_RUN_LOG=str(logs["run"]),
        FAKE_RC=str(fake_rc),
        FAKE_TASK=f"{RUN}\\t{bench}\\t{task_host}\\t{RUN_DIR}\\t{opts}" if task else "",
        SAVES_DIR=str(saves),
        DIST_MASTER="masterhost",
        DIST_MASTER_REPO="/repo",
        DIST_HOSTS="hostB=/remote/saves hostC=/other/saves",
        DIST_SELF_HOST="hostC",
        DIST_RUN_SH=str(run_sh),
        DIST_ONCE="1",
        DIST_SKIP_AGENT="1",
    )
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "dist_worker.sh"), gpus],
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    read = {k: p.read_text().splitlines() for k, p in logs.items()}
    return result, read, saves


def test_remote_task_is_fetched_evaluated_reported_and_its_checkpoint_deleted(tmp_path):
    result, logs, saves = _run_worker(tmp_path)
    assert result.returncode == 0, result.stderr

    assert len(logs["rsync"]) == 1
    rsync = logs["rsync"][0]
    assert f"hostB:/remote/saves/train_logs/{RUN_DIR}/./.hydra" in rsync
    assert f"hostB:/remote/saves/train_logs/{RUN_DIR}/./seed_*/saved_models/last.ckpt" in rsync
    assert "eval_logs" not in rsync
    # a training host may have moved .hydra or the checkpoint elsewhere and left a symlink:
    # copy what it points to, not a link that only resolves on that host
    assert "--copy-links" in rsync.split()

    assert logs["run"] == [f"CVD=1 RESUME=1 MULTI=1 ARGS=pipeline /saves/train_logs/{RUN_DIR}"]
    assert any("masterhost" in l and "dist-sched" in l and "claim" in l and "hostC:1" in l for l in logs["ssh"])
    assert any("dist-sched" in l and f"done --run {RUN} --worker hostC:1" in l for l in logs["ssh"])

    local = saves / "train_logs" / RUN_DIR
    assert not (local / "seed_42" / "saved_models" / "last.ckpt").exists()
    assert (local / ".hydra" / "config.yaml").exists()
    assert (saves / "dist_logs" / f"{RUN}.log").exists()


def test_local_task_is_neither_copied_nor_deleted(tmp_path):
    local = tmp_path / "saves" / "train_logs" / RUN_DIR / "seed_42" / "saved_models"
    local.mkdir(parents=True)
    (local / "last.ckpt").write_text("ckpt")

    result, logs, _ = _run_worker(tmp_path, task_host="hostC")
    assert result.returncode == 0, result.stderr
    assert logs["rsync"] == []
    assert len(logs["run"]) == 1
    assert (local / "last.ckpt").exists()
    assert any(f"done --run {RUN}" in l for l in logs["ssh"])
    # the worker's own runs need no ssh except to the master
    assert all(l.split()[0] == "masterhost" or "-o" in l.split()[0] for l in logs["ssh"]), logs["ssh"]


@pytest.mark.parametrize("bench, sub", [("calvin", "pipeline-calvin"), ("mimicgen", "pipeline-mimicgen")])
def test_benchmark_selects_its_pipeline(tmp_path, bench, sub):
    result, logs, _ = _run_worker(tmp_path, opts="", bench=bench)
    assert result.returncode == 0, result.stderr
    assert logs["run"] == [f"CVD=1 RESUME=1 MULTI= ARGS={sub} /saves/train_logs/{RUN_DIR}"]


def test_failed_pipeline_is_reported_and_checkpoint_still_deleted(tmp_path):
    result, logs, saves = _run_worker(tmp_path, fake_rc=1)
    assert result.returncode == 0, result.stderr
    assert any(f"fail --run {RUN} --worker hostC:1" in l for l in logs["ssh"])
    assert not any(" done " in l for l in logs["ssh"])
    assert not (saves / "train_logs" / RUN_DIR / "seed_42" / "saved_models" / "last.ckpt").exists()


def test_unknown_training_host_fails_the_task_without_running(tmp_path):
    result, logs, _ = _run_worker(tmp_path, task_host="hostZ")
    assert result.returncode == 0, result.stderr
    assert logs["run"] == [] and logs["rsync"] == []
    assert any(f"fail --run {RUN}" in l and "hostZ" in l for l in logs["ssh"])


def test_no_task_means_no_work(tmp_path):
    result, logs, _ = _run_worker(tmp_path, task=False)
    assert result.returncode == 0, result.stderr
    assert logs["run"] == [] and logs["rsync"] == []
    assert sum("claim" in l for l in logs["ssh"]) == 1


def test_one_loop_per_gpu(tmp_path):
    result, logs, _ = _run_worker(tmp_path, gpus="0,3")
    assert result.returncode == 0, result.stderr
    claims = sorted(l for l in logs["ssh"] if "claim" in l)
    assert len(claims) == 2
    assert any("hostC:0" in l for l in claims) and any("hostC:3" in l for l in claims)
    assert len(logs["run"]) == 1  # the fake master hands out one task


# ---------------------------------------------------------------------------
# Stopping the worker (Ctrl-C / closed window) releases its tasks
# ---------------------------------------------------------------------------

STUB_PODMAN = """#!/usr/bin/env bash
echo "$*" >> "$FAKE_PODMAN_LOG"
case "$1" in
  ps) printf '%s\\n' "c0ffee python flower/evaluation/flower_eval_libero.py train_folder=/saves/train_logs/$FAKE_RUN_DIR x=1" \\
                       "beef00 python flower/evaluation/flower_eval_libero.py train_folder=/saves/train_logs/other/run" ;;
esac
"""

STUB_SLOW_RUN_SH = """#!/usr/bin/env bash
echo $$ > "$FAKE_RUN_LOG.pid"
echo "started" >> "$FAKE_RUN_LOG"
sleep 60
echo "finished" >> "$FAKE_RUN_LOG"
"""


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_stopping_the_worker_releases_its_task_and_stops_its_containers(tmp_path, sig):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub(bin_dir / "ssh", STUB_SSH)
    _stub(bin_dir / "rsync", STUB_RSYNC)
    _stub(bin_dir / "podman", STUB_PODMAN)
    run_sh = tmp_path / "run.sh"
    _stub(run_sh, STUB_SLOW_RUN_SH)
    saves = tmp_path / "saves"
    saves.mkdir()
    logs = {k: tmp_path / f"{k}.log" for k in ("ssh", "rsync", "run", "podman")}
    for p in logs.values():
        p.write_text("")
    env = dict(os.environ)
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}",
        FAKE_SSH_LOG=str(logs["ssh"]),
        FAKE_RSYNC_LOG=str(logs["rsync"]),
        FAKE_RUN_LOG=str(logs["run"]),
        FAKE_PODMAN_LOG=str(logs["podman"]),
        FAKE_RUN_DIR=RUN_DIR,
        FAKE_TASK=f"{RUN}\\tlibero\\thostB\\t{RUN_DIR}\\t",
        SAVES_DIR=str(saves),
        DIST_MASTER="masterhost",
        DIST_MASTER_REPO="/repo",
        DIST_HOSTS="hostB=/remote/saves",
        DIST_SELF_HOST="hostC",
        DIST_RUN_SH=str(run_sh),
        DIST_SKIP_AGENT="1",
        DIST_POLL_MIN="10",
    )
    worker = subprocess.Popen(
        ["bash", str(REPO_ROOT / "scripts" / "dist_worker.sh"), "1"],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.time() + 20
    while "started" not in logs["run"].read_text() and time.time() < deadline:
        time.sleep(0.1)
    assert "started" in logs["run"].read_text(), "the fake pipeline never started"
    pipeline_pid = int(Path(str(logs["run"]) + ".pid").read_text())

    worker.send_signal(sig)
    out, err = worker.communicate(timeout=30)
    assert worker.returncode == 0, err

    ssh = logs["ssh"].read_text()
    assert f"release --run {RUN} --worker hostC:1" in ssh
    assert " fail " not in ssh and " done " not in ssh
    podman = logs["podman"].read_text().splitlines()
    assert any(l.startswith("stop") and "c0ffee" in l and "beef00" not in l for l in podman), podman
    with pytest.raises(ProcessLookupError):
        os.kill(pipeline_pid, 0)  # the pipeline itself was stopped
    assert "finished" not in logs["run"].read_text()
    assert not (saves / "train_logs" / RUN_DIR / "seed_42" / "saved_models" / "last.ckpt").exists()


def test_stopping_an_idle_worker_exits_promptly(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub(bin_dir / "ssh", STUB_SSH)
    env = dict(os.environ)
    ssh_log = tmp_path / "ssh.log"
    ssh_log.write_text("")
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}",
        FAKE_SSH_LOG=str(ssh_log),
        FAKE_TASK="",
        SAVES_DIR=str(tmp_path),
        DIST_MASTER="masterhost",
        DIST_MASTER_REPO="/repo",
        DIST_SELF_HOST="hostC",
        DIST_SKIP_AGENT="1",
        DIST_POLL_MIN="10",
    )
    worker = subprocess.Popen(
        ["bash", str(REPO_ROOT / "scripts" / "dist_worker.sh"), "0"],
        env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    deadline = time.time() + 20
    while "claim" not in ssh_log.read_text() and time.time() < deadline:
        time.sleep(0.1)
    start = time.time()
    worker.send_signal(signal.SIGINT)
    worker.communicate(timeout=20)
    assert time.time() - start < 10  # does not wait out the 10-minute poll sleep
    assert "release" not in ssh_log.read_text()
