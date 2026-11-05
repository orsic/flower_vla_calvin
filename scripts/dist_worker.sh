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

# Evaluate one claimed task on one GPU; always reports done or fail to the master.
run_task() {
    local gpu="$1" line="$2" run_id bench host run_dir opts worker local_dir rs sub rc=0 hb
    IFS=$'\t' read -r run_id bench host run_dir opts <<< "$line"
    worker="$SELF:$gpu"
    local_dir="$SAVES_DIR/train_logs/$run_dir"
    mkdir -p "$SAVES_DIR/dist_logs"
    local tlog="$SAVES_DIR/dist_logs/$run_id.log"
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
        if ! rs="$(remote_saves "$host")"; then
            rc=1; log "host $host is not in DIST_HOSTS" | tee -a "$tlog"
        else
            mkdir -p "$local_dir"
            rsync -a --relative \
                "$host:$rs/train_logs/$run_dir/./.hydra" \
                "$host:$rs/train_logs/$run_dir/./seed_*/saved_models/last.ckpt" \
                "$local_dir/" >> "$tlog" 2>&1 || { rc=$?; log "rsync from $host failed ($rc)" | tee -a "$tlog"; }
        fi
    fi

    if [[ $rc -eq 0 ]]; then
        # shellcheck disable=SC2086  # opts is a list of VAR=VALUE words
        env CUDA_VISIBLE_DEVICES="$gpu" PIPELINE_RESUME=1 $opts \
            "$RUN_SH" "$sub" "/saves/train_logs/$run_dir" >> "$tlog" 2>&1 < /dev/null || rc=$?
    fi

    kill "$hb" 2>/dev/null
    wait "$hb" 2>/dev/null
    if [[ $rc -eq 0 ]]; then
        master done --run "$run_id" --worker "$worker" > /dev/null || log "master refused 'done' for $run_id"
        log "GPU $gpu: $run_id done" | tee -a "$tlog"
    else
        local msg="exit $rc on $worker"
        [[ -n "$host" ]] && ! remote_saves "$host" > /dev/null && [[ "$host" != "$SELF" ]] && msg="unknown host $host on $worker"
        master fail --run "$run_id" --worker "$worker" --msg "$msg" > /dev/null || log "master refused 'fail' for $run_id"
        log "GPU $gpu: $run_id FAILED ($msg), see $tlog" | tee -a "$tlog"
    fi
    if [[ $remote -eq 1 ]]; then
        rm -f "$local_dir"/seed_*/saved_models/last.ckpt
    fi
}

gpu_loop() {
    local gpu="$1" line
    while true; do
        if ! line="$(master claim --worker "$SELF:$gpu")"; then
            log "GPU $gpu: could not reach the master's queue"
            line=""
        fi
        if [[ -n "$line" ]]; then
            run_task "$gpu" "$line"
        fi
        [[ -n "${DIST_ONCE:-}" ]] && return 0
        [[ -z "$line" ]] && sleep "$POLL_S"
    done
}

[[ -z "${DIST_SKIP_AGENT:-}" ]] && setup_agent
log "worker $SELF: GPUs ${GPUS[*]}, master $DIST_MASTER"
trap 'log "stopping"; kill $(jobs -p) 2>/dev/null; exit 0' INT TERM
for gpu in "${GPUS[@]}"; do
    gpu_loop "$gpu" &
done
wait
