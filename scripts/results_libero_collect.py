"""Download the latest evaluation artifact of every LIBERO run in both W&B projects and
write an inventory (config, host, args, artifact files, eval timestamp ranges) to
<out>/inventory.json. scripts/results_libero_tables.py reads the downloaded files.

    python scripts/results_libero_collect.py [--out /saves/libero_artifacts]

Needs WANDB_API_KEY.
"""

import argparse
import csv
import json
from pathlib import Path

import wandb

ENTITY = "orsic-university-of-zagreb-faculty-of-electrical-enginee"
PROJECTS = ["multimodal_florence", "multimodal_policies"]
KEYS = [
    "use_proprio",
    "modality_dropout",
    "modality_dropout_keep_fraction",
    "modality_dropout_proprio_keep_p",
    "modality_dropout_alphas",
    "modalities",
    "n_layers",
    "pretrained_dit_layers",
    "extra_layer_init",
    "extra_layer_placement",
    "freeze_florence",
    "action_expert_from_scratch",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=Path("/saves/libero_artifacts"))
    out = ap.parse_args().out

    api = wandb.Api(timeout=120)
    inv = []
    for project in PROJECTS:
        for run in api.runs(f"{ENTITY}/{project}"):
            if "libero" not in run.id and "libero" not in (run.group or "") and "libero" not in run.name:
                continue
            meta = run.metadata or {}
            rec = {
                "project": project,
                "id": run.id,
                "name": run.name,
                "group": run.group,
                "state": run.state,
                "created": run.created_at,
                "heartbeat": str(run.heartbeat_at),
                "host": meta.get("host"),
                "args": meta.get("args"),
                "config": {k: run.config.get(k) for k in KEYS},
                "epoch": run.summary.get("epoch"),
                "avg_seq_len": run.summary.get("eval_lh/avg_seq_len"),
                "artifact": None,
                "files": {},
            }
            evals = [a for a in run.logged_artifacts() if a.type == "evaluation"]
            if evals:
                art = max(evals, key=lambda a: a.created_at)
                d = Path(art.download(root=str(out / project / run.id)))
                rec["artifact"] = {"name": art.name, "created": art.created_at, "versions": len(evals)}
                for f in sorted(d.iterdir()):
                    info = {"bytes": f.stat().st_size}
                    if f.suffix == ".csv":
                        rows = list(csv.DictReader(open(f)))
                        info["rows"] = len(rows)
                        ts = sorted(r["eval_timestamp"] for r in rows if r.get("eval_timestamp"))
                        if ts:
                            info["ts"] = [ts[0], ts[-1]]
                        ck = sorted({r["checkpoint"] for r in rows if r.get("checkpoint")})
                        if ck:
                            info["checkpoints"] = ck
                    rec["files"][f.name] = info
            inv.append(rec)
            print(project, run.id, run.state, rec["artifact"]["name"] if rec["artifact"] else "-", flush=True)
    (out / "inventory.json").write_text(json.dumps(inv, indent=1, default=str))
    print("runs", len(inv))


if __name__ == "__main__":
    main()
