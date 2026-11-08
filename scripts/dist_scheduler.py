"""Distributed-evaluation task queue, kept on the master machine.

Finished training runs are discovered on W&B; evaluation workers (scripts/dist_worker.sh,
started with `./run.sh dist-worker <gpus>` on any machine) claim them one per GPU over
SSH, evaluate them with the benchmark's pipeline, and report back. The queue lives in
one JSON file guarded by an exclusive lock, so concurrent claims never hand out the
same run twice.

A run is queued when it started on or after --since (the local date in its name, e.g.
calvin_abcd_d_2026-11-06_00-00-18; W&B's created_at is UTC), finished training (W&B state
"finished" and its last epoch reached max_epochs - 1, max_epochs from the benchmark's
conf/config_*.yaml or a max_epochs= override in the run's args), and has no
"evaluation" artifact yet. A running task's lease expires --lease-min minutes after its
last heartbeat, which returns it to the queue; a failed evaluation is retried once,
then marked failed.

    ./run.sh dist-sched refresh | status | claim --worker H:G | heartbeat --run R --worker H:G
    ./run.sh dist-sched done|fail|release --run R --worker H:G [--msg M] | requeue --run R | skip --run R
"""

import argparse
import contextlib
import fcntl
import fnmatch
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).absolute().parents[1]
BENCHMARK_PREFIXES = (("libero_", "libero"), ("calvin_", "calvin"), ("mimicgen", "mimicgen"))
CONFIGS = {
    "libero": "conf/config_libero.yaml",
    "calvin": "conf/config_calvin.yaml",
    "mimicgen": "conf/config_mimicgen.yaml",
}
MAX_ATTEMPTS = 2
REFRESH_EVERY = timedelta(minutes=5)
RUN_DIR_ARG = "hydra.run.dir=/saves/train_logs/"

Rules = List[Tuple[str, str]]


def benchmark_of(run_id: str) -> Optional[str]:
    for prefix, bench in BENCHMARK_PREFIXES:
        if run_id.startswith(prefix):
            return bench
    return None


def max_epochs(benchmark: str, args: Iterable[str]) -> int:
    """The benchmark config's max_epochs, unless the run's args override it."""
    value = yaml.safe_load((REPO_ROOT / CONFIGS[benchmark]).read_text())["max_epochs"]
    for arg in args:
        if arg.startswith("max_epochs="):
            value = arg.split("=", 1)[1]
    return int(value)


def parse_opts(spec: str) -> Rules:
    """'glob:VAR=VAL VAR2=VAL;glob2:...' -> [(glob, 'VAR=VAL VAR2=VAL'), ...]."""
    rules = []
    for part in spec.split(";"):
        if not part.strip():
            continue
        glob, _, opts = part.partition(":")
        for token in opts.split():
            if "=" not in token:
                raise ValueError(f"DIST_EVAL_OPTS: {token!r} in {part!r} is not VAR=VALUE")
        rules.append((glob.strip(), " ".join(opts.split())))
    return rules


def opts_for(run_id: str, rules: Rules) -> str:
    for glob, opts in rules:
        if fnmatch.fnmatch(run_id, glob):
            return opts
    return ""


def run_date(run) -> str:
    """The local start time in the run's name (YYYY-MM-DD_HH-MM-SS), else W&B's UTC
    created_at. DIST_SINCE compares against this, so a run started just after local
    midnight counts for that day."""
    m = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", run.id)
    return m.group() if m else run.created_at


def _evaluated(run) -> bool:
    return any(a.type == "evaluation" for a in run.logged_artifacts())


def task_from_run(run, since: str, rules: Rules) -> Optional[Dict]:
    """A new todo task for a finished, not yet evaluated run, else None."""
    bench = benchmark_of(run.id)
    if bench is None or run_date(run) < since or run.state != "finished":
        return None
    meta = run.metadata or {}
    args = meta.get("args") or []
    run_dir = next((a[len(RUN_DIR_ARG) :] for a in args if a.startswith(RUN_DIR_ARG)), None)
    epoch = run.summary.get("epoch")
    if run_dir is None or epoch is None or epoch < max_epochs(bench, args) - 1:
        return None
    if _evaluated(run):
        return None
    return {
        "run_id": run.id,
        "benchmark": bench,
        "host": meta.get("host"),
        "run_dir": run_dir,
        "opts": opts_for(run.id, rules),
        "created": run.created_at,
        "state": "todo",
        "worker": None,
        "attempts": 0,
        "claimed": None,
        "heartbeat": None,
        "finished": None,
        "error": None,
    }


def empty_state() -> Dict:
    return {"refreshed": None, "tasks": {}}


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def refresh(state: Dict, runs: Iterable, since: str, rules: Rules, now: datetime) -> None:
    """Queue newly finished runs; a queued todo run evaluated elsewhere meanwhile is done."""
    tasks = state["tasks"]
    for run in runs:
        if run.id in tasks:
            task = tasks[run.id]
            if task["state"] == "todo" and _evaluated(run):
                task.update(state="done", finished=_iso(now), error="evaluated outside the scheduler")
            continue
        task = task_from_run(run, since, rules)
        if task is not None:
            tasks[run.id] = task
    state["refreshed"] = _iso(now)


def expire_leases(state: Dict, now: datetime, lease_min: float) -> None:
    for task in state["tasks"].values():
        if task["state"] == "running" and now - _parse(task["heartbeat"]) > timedelta(minutes=lease_min):
            task.update(state="todo", worker=None, error=f"lease expired ({task['worker']})")


def claim(state: Dict, worker: str, now: datetime) -> Optional[Dict]:
    todo = [t for t in state["tasks"].values() if t["state"] == "todo"]
    if not todo:
        return None
    task = min(todo, key=lambda t: t["created"])
    task.update(state="running", worker=worker, claimed=_iso(now), heartbeat=_iso(now))
    task["attempts"] += 1
    return task


def _owned(state: Dict, run_id: str, worker: str) -> Optional[Dict]:
    task = state["tasks"].get(run_id)
    if task is None or task["state"] != "running" or task["worker"] != worker:
        return None
    return task


def heartbeat(state: Dict, run_id: str, worker: str, now: datetime) -> bool:
    task = _owned(state, run_id, worker)
    if task is not None:
        task["heartbeat"] = _iso(now)
    return task is not None


def finish(state: Dict, run_id: str, worker: str, ok: bool, msg: str, now: datetime) -> bool:
    task = _owned(state, run_id, worker)
    if task is None:
        return False
    if ok:
        task.update(state="done", finished=_iso(now), error=None)
    elif task["attempts"] < MAX_ATTEMPTS:
        task.update(state="todo", worker=None, error=msg)
    else:
        task.update(state="failed", finished=_iso(now), error=msg)
    return True


def release(state: Dict, run_id: str, worker: str, now: Optional[datetime] = None) -> bool:
    """The worker stopped (Ctrl-C) before finishing: back to todo at once, and the
    interrupted attempt does not count towards MAX_ATTEMPTS."""
    task = _owned(state, run_id, worker)
    if task is None:
        return False
    task.update(state="todo", worker=None, attempts=max(task["attempts"] - 1, 0), error=f"released by {worker}")
    return True


def requeue(state: Dict, run_id: str) -> None:
    state["tasks"][run_id].update(state="todo", worker=None, attempts=0, error=None)


def skip(state: Dict, run_id: str) -> None:
    state["tasks"][run_id].update(state="skipped", worker=None)


def task_line(task: Dict) -> str:
    """What `claim` prints for the worker: tab-separated, no JSON parsing needed in bash."""
    return "\t".join([task["run_id"], task["benchmark"], task["host"], task["run_dir"], task["opts"]])


def _age(stamp: Optional[str], now: datetime) -> str:
    if not stamp:
        return "-"
    minutes = int((now - _parse(stamp)).total_seconds() // 60)
    return f"{minutes // 60}h{minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


def status_table(state: Dict, now: datetime) -> str:
    tasks = sorted(state["tasks"].values(), key=lambda t: t["created"])
    counts = {}
    for t in tasks:
        counts[t["state"]] = counts.get(t["state"], 0) + 1
    order = ("todo", "running", "done", "failed", "skipped")
    lines = [
        "refreshed: " + (state["refreshed"] or "never"),
        "  ".join(f"{s} {counts.get(s, 0)}" for s in order),
        "",
    ]
    header = ("run", "bench", "host", "state", "worker", "tries", "since", "heartbeat", "opts / error")
    rows = [header]
    for t in tasks:
        since = t["finished"] if t["state"] in ("done", "failed") else t["claimed"]
        note = t["error"] or t["opts"] or ""
        rows.append(
            (
                t["run_id"],
                t["benchmark"],
                t["host"] or "-",
                t["state"],
                t["worker"] or "-",
                str(t["attempts"]),
                _age(since, now),
                _age(t["heartbeat"], now) if t["state"] == "running" else "-",
                note,
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(header) - 1)]
    for r in rows:
        lines.append("  ".join(c.ljust(w) for c, w in zip(r, widths)) + "  " + r[-1])
    return "\n".join(lines)


@contextlib.contextmanager
def locked_state(path: Path) -> Iterator[Dict]:
    """Read-modify-write of the state file under an exclusive lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_text()) if path.exists() else empty_state()
        yield state
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        os.replace(tmp, path)


def _wandb_runs(since: str) -> List:
    import wandb

    logger = yaml.safe_load((REPO_ROOT / CONFIGS["libero"]).read_text())["logger"]
    api = wandb.Api(timeout=120)
    # Query a day early: created_at is UTC, task_from_run applies --since to the local date.
    day_before = (datetime.strptime(since[:10], "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    return list(api.runs(f"{logger['entity']}/{logger['project']}", filters={"created_at": {"$gte": day_before}}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--state", type=Path, default=Path("/saves/dist_scheduler/state.json"))
    ap.add_argument("--since", default="")
    ap.add_argument("--opts", default="")
    ap.add_argument("--lease-min", type=float, default=30)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("refresh")
    sub.add_parser("status")
    for name in ("claim", "heartbeat", "done", "fail", "release", "requeue", "skip"):
        p = sub.add_parser(name)
        if name != "claim":
            p.add_argument("--run", required=True)
        if name in ("claim", "heartbeat", "done", "fail", "release"):
            p.add_argument("--worker", required=True)
        if name in ("done", "fail"):
            p.add_argument("--msg", default="")
    args = ap.parse_args()
    now = datetime.now(timezone.utc)

    if args.cmd in ("refresh", "claim", "status") and not args.since:
        sys.exit("dist_scheduler: DIST_SINCE is not set (vars.env on the master)")
    rules = parse_opts(args.opts)

    with locked_state(args.state) as state:
        stale = state["refreshed"] is None or now - _parse(state["refreshed"]) > REFRESH_EVERY
        if args.cmd == "refresh" or (args.cmd in ("claim", "status") and stale):
            refresh(state, _wandb_runs(args.since), args.since, rules, now)
        expire_leases(state, now, args.lease_min)

        if args.cmd == "claim":
            task = claim(state, args.worker, now)
            if task is not None:
                print(task_line(task))
        elif args.cmd == "heartbeat":
            if not heartbeat(state, args.run, args.worker, now):
                sys.exit(3)
        elif args.cmd in ("done", "fail"):
            if not finish(state, args.run, args.worker, args.cmd == "done", args.msg, now):
                sys.exit(3)
        elif args.cmd == "release":
            if not release(state, args.run, args.worker):
                sys.exit(3)
        elif args.cmd == "requeue":
            requeue(state, args.run)
        elif args.cmd == "skip":
            skip(state, args.run)
        elif args.cmd in ("status", "refresh"):
            print(status_table(state, now))


if __name__ == "__main__":
    main()
