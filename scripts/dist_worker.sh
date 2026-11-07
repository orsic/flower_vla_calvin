#!/usr/bin/env bash
# Distributed-evaluation worker (./run.sh dist-worker <gpu,gpu,...>).
#
# One loop per GPU: claim a task from the master's queue (scripts/dist_scheduler.py, over
# ssh), fetch the run's .hydra config and checkpoint from the host it trained on (rsync
# over ssh, into the same train_logs/<group>/<ts> path under this machine's SAVES_DIR),
# run the benchmark's pipeline on that GPU with the task's options, report done/fail,
# and delete the copied checkpoint. A run that trained on this machine is evaluated in
# place: no ssh, no copy, no deletion. Loops run until the worker is killed (Ctrl-C).
#
# Stopping the worker (Ctrl-C, kill, closing the window) stops every running pipeline and
# the podman containers it started, hands the task back to the queue (`dist-sched
# release`, not counted as a failed attempt) and deletes a copied checkpoint.
#
# Startup starts an ssh-agent for this process tree only (i.e. this tmux window) and
# adds the IdentityFiles `ssh -G` reports for the master and every DIST_HOSTS host, so
# passphrases are asked once and no other shell gets the keys.
#
# vars.env (never committed):
#   DIST_MASTER=<host>             DIST_MASTER_REPO=<repo path on the master>
#   DIST_HOSTS="hostA=/path/to/saves hostB=/path/to/saves"   training hosts + their SAVES_DIR
#   DIST_POLL_MIN=10 (idle GPU re-asks)   DIST_HEARTBEAT_MIN=5
# Test knobs: DIST_SELF_HOST, DIST_RUN_SH, DIST_ONCE=1 (one claim per GPU), DIST_SKIP_AGENT=1.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SH="${DIST_RUN_SH:-$REPO_ROOT/run.sh}"
SELF="${DIST_SELF_HOST:-${HOST_HOSTNAME:-$(uname -n)}}"
POLL_S=$(( ${DIST_POLL_MIN:-10} * 60 ))
HEARTBEAT_S=$(( ${DIST_HEARTBEAT_MIN:-5} * 60 ))
: "${SAVES_DIR:?SAVES_DIR is not set (vars.env)}"
: "${DIST_MASTER:?DIST_MASTER is not set (vars.env)}"
: "${DIST_MASTER_REPO:?DIST_MASTER_REPO is not set (vars.env)}"
DIST_HOSTS="${DIST_HOSTS:-}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20)

if [[ $# -ne 1 || -z "$1" ]]; then
    echo "Usage: ./run.sh dist-worker <gpu[,gpu,...]>" >&2
    exit 1
fi
IFS=',' read -r -a GPUS <<< "$1"

log() { echo "[$(date '+%F %T')] $*"; }

# Run `./run.sh dist-sched <args>` on the master (locally when this is the master).
master() {
    if [[ "$DIST_MASTER" == "$SELF" ]]; then
        (cd "$DIST_MASTER_REPO" && ./run.sh dist-sched "$@")
    else
        "${SSH[@]}" "$DIST_MASTER" "cd $(printf %q "$DIST_MASTER_REPO") && ./run.sh dist-sched $(printf '%q ' "$@")"
    fi
}

remote_saves() {
    local entry
    for entry in $DIST_HOSTS; do
        [[ "${entry%%=*}" == "$1" ]] && { echo "${entry#*=}"; return 0; }
    done
    return 1
}

setup_agent() {
    local hosts=() keys=() h f entry
    [[ "$DIST_MASTER" != "$SELF" ]] && hosts+=("$DIST_MASTER")
    for entry in $DIST_HOSTS; do
        h="${entry%%=*}"
        [[ "$h" != "$SELF" && " ${hosts[*]} " != *" $h "* ]] && hosts+=("$h")
    done
    [[ ${#hosts[@]} -eq 0 ]] && return 0
    eval "$(ssh-agent -s)" > /dev/null
    trap 'ssh-agent -k > /dev/null 2>&1' EXIT
    for h in "${hosts[@]}"; do
        while read -r f; do
            f="${f/#\~/$HOME}"
            [[ -f "$f" && " ${keys[*]} " != *" $f "* ]] && keys+=("$f")
        done < <(ssh -G "$h" | awk '$1 == "identityfile" {print $2}')
    done
    if [[ ${#keys[@]} -gt 0 ]]; then
        log "unlocking ${#keys[@]} key(s) for: ${hosts[*]}"
        ssh-add "${keys[@]}" || { echo "ssh-add failed" >&2; exit 1; }
    fi
    for h in "${hosts[@]}"; do
        "${SSH[@]}" "$h" true || log "WARNING: cannot reach $h over ssh"
    done
}

# Per-GPU-loop state (each loop is its own subshell), read by on_stop when the worker is
# stopped mid-task.
WORKER="" CUR_RUN="" CUR_DIR="" CUR_PID="" HB_PID="" CUR_REMOTE=0

# Stop the podman containers whose command names this run's folder: the pipeline's eval,
# planner and upload containers all take it as train_folder / --train-folder.
stop_containers() {
    podman ps --no-trunc --format '{{.ID}} {{.Command}}' 2>/dev/null \
        | grep -F "/saves/train_logs/$1" | awk '{print $1}' \
        | xargs -r podman stop -t 10 > /dev/null 2>&1
}

on_stop() {
    if [[ -n "$CUR_PID" ]]; then kill "$CUR_PID" 2>/dev/null; fi  # first, so it starts nothing new
    if [[ -n "$HB_PID" ]]; then kill "$HB_PID" 2>/dev/null; fi
    if [[ -n "$CUR_RUN" ]]; then
        stop_containers "$CUR_DIR"
        master release --run "$CUR_RUN" --worker "$WORKER" > /dev/null 2>&1 \
            || log "master refused 'release' for $CUR_RUN"
        if [[ $CUR_REMOTE -eq 1 ]]; then
            rm -f "$SAVES_DIR/train_logs/$CUR_DIR"/seed_*/saved_models/last.ckpt
        fi
        log "$WORKER: stopped, released $CUR_RUN"
    fi
    exit 0
}

# Evaluate one claimed task on one GPU; reports done or fail to the master (or, if the
# worker is stopped meanwhile, on_stop releases it).
run_task() {
    local gpu="$1" line="$2" run_id bench host run_dir opts worker local_dir rs sub rc=0 hb
    IFS=$'\t' read -r run_id bench host run_dir opts <<< "$line"
    worker="$SELF:$gpu"
    local_dir="$SAVES_DIR/train_logs/$run_dir"
    mkdir -p "$SAVES_DIR/dist_logs"
    local tlog="$SAVES_DIR/dist_logs/$run_id.log"
    CUR_RUN="$run_id" CUR_DIR="$run_dir" CUR_REMOTE=0
    log "GPU $gpu: $run_id ($bench, trained on $host)" | tee -a "$tlog"

    # Heartbeat while the task runs. The sleep is waited on (not run in the foreground) so
    # the TERM sent below interrupts it at once and the trap kills it; no orphan keeps
    # this worker's stdout open.
    (
        trap 'kill "$s" 2>/dev/null; exit 0' TERM
        while true; do
            sleep "$HEARTBEAT_S" & s=$!
            wait "$s"
            master heartbeat --run "$run_id" --worker "$worker"
        done
    ) > /dev/null 2>&1 < /dev/null &
    hb=$!
    HB_PID=$hb

    case "$bench" in
        libero) sub=pipeline ;;
        calvin) sub=pipeline-calvin ;;
        mimicgen) sub=pipeline-mimicgen ;;
        *) sub="" ;;
    esac

    local remote=0
    if [[ -z "$sub" ]]; then
        rc=1; opts=""; log "unknown benchmark $bench" >> "$tlog"
    elif [[ "$host" != "$SELF" ]]; then
        remote=1
        CUR_REMOTE=1
        if ! rs="$(remote_saves "$host")"; then
            rc=1; log "host $host is not in DIST_HOSTS" | tee -a "$tlog"
        else
            mkdir -p "$local_dir"
            # Background + wait (here and for the pipeline below): a stop signal runs
            # on_stop right away instead of after the child finishes.
            rsync -a --relative \
                "$host:$rs/train_logs/$run_dir/./.hydra" \
                "$host:$rs/train_logs/$run_dir/./seed_*/saved_models/last.ckpt" \
                "$local_dir/" >> "$tlog" 2>&1 &
            CUR_PID=$!
            wait "$CUR_PID" || { rc=$?; log "rsync from $host failed ($rc)" | tee -a "$tlog"; }
        fi
    fi

    if [[ $rc -eq 0 ]]; then
        # shellcheck disable=SC2086  # opts is a list of VAR=VALUE words
        env CUDA_VISIBLE_DEVICES="$gpu" PIPELINE_RESUME=1 $opts \
            "$RUN_SH" "$sub" "/saves/train_logs/$run_dir" >> "$tlog" 2>&1 < /dev/null &
        CUR_PID=$!
        wait "$CUR_PID" || rc=$?
    fi
    CUR_PID=""

    kill "$hb" 2>/dev/null
    wait "$hb" 2>/dev/null
    HB_PID=""
    if [[ $rc -eq 0 ]]; then
        master done --run "$run_id" --worker "$worker" > /dev/null || log "master refused 'done' for $run_id"
        log "GPU $gpu: $run_id done" | tee -a "$tlog"
    else
        local msg="exit $rc on $worker"
        [[ -n "$host" ]] && ! remote_saves "$host" > /dev/null && [[ "$host" != "$SELF" ]] && msg="unknown host $host on $worker"
        master fail --run "$run_id" --worker "$worker" --msg "$msg" > /dev/null || log "master refused 'fail' for $run_id"
        log "GPU $gpu: $run_id FAILED ($msg), see $tlog" | tee -a "$tlog"
    fi
    CUR_RUN=""
    if [[ $remote -eq 1 ]]; then
        rm -f "$local_dir"/seed_*/saved_models/last.ckpt
    fi
}

gpu_loop() {
    local gpu="$1" line
    WORKER="$SELF:$gpu"
    trap on_stop TERM
    while true; do
        if ! line="$(master claim --worker "$SELF:$gpu")"; then
            log "GPU $gpu: could not reach the master's queue"
            line=""
        fi
        if [[ -n "$line" ]]; then
            run_task "$gpu" "$line"
        fi
        [[ -n "${DIST_ONCE:-}" ]] && return 0
        if [[ -z "$line" ]]; then
            sleep "$POLL_S" &
            CUR_PID=$!
            wait "$CUR_PID"
            CUR_PID=""
        fi
    done
}

[[ -z "${DIST_SKIP_AGENT:-}" ]] && setup_agent
log "worker $SELF: GPUs ${GPUS[*]}, master $DIST_MASTER"
# Background loops ignore Ctrl-C (SIGINT), so this trap is what stops them: each loop's
# own TERM trap (on_stop) stops its pipeline and releases its task before it exits.
LOOPS=()
trap 'log "stopping"; kill -TERM "${LOOPS[@]}" 2>/dev/null; wait; exit 0' INT TERM HUP
for gpu in "${GPUS[@]}"; do
    gpu_loop "$gpu" &
    LOOPS+=($!)
done
wait
