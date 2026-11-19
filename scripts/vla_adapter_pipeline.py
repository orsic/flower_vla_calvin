#!/usr/bin/env python3
"""Plan, and later collect, VLA-Adapter's evals with one input withheld, the way
scripts/eval_pipeline.py and scripts/calvin_pipeline.py do for a FlowerVLA training run.

A VLA-Adapter checkpoint has no training run (.hydra/config.yaml), so the FlowerVLA
planners can't resolve it; this planner reuses their pieces instead: all inputs, each
input removed (eval_pipeline.modality_overrides) and each input masked
(eval_pipeline.masked_lines). The evals write FlowerVLA's result layout under the
checkpoint dir (train_folder = checkpoint): removed inputs merge into
<ckpt>/eval_logs/<ckpt>/<suite_dir>/result.csv by their use_* flags, masked inputs go to
<ckpt>/eval_logs/<ckpt>/<variant>/<suite_dir>/result.csv. Driven by
./run.sh pipeline-vla-adapter.

Usage:
  python scripts/vla_adapter_pipeline.py plan --benchmark libero|calvin [--ckpt-root DIR] [--resume]
  python scripts/vla_adapter_pipeline.py collect --benchmark libero|calvin [--ckpt-root DIR]

`plan` prints one line per eval, tab-separated:
    <eval script>\t<override1>\t<override2>\t...
`collect` copies every result.csv to <ckpt>/artifact/ under the artifact member names the
FlowerVLA pipelines upload (libero_orig.csv / calvin.csv, <variant>/<member>), so the
analysis code that reads downloaded artifacts reads these unchanged.
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from eval_pipeline import combo_in_csv, masked_artifact_members, masked_lines, modality_overrides  # noqa: E402

DEFAULT_CKPT_ROOT = "/saves/checkpoints/vla_adapter"
FULL = {"rgb_static": True, "rgb_gripper": True, "language": True, "proprio": True}
BENCHMARKS = {
    # LIBERO: FlowerVLA's LIBERO-10 evals cover episodes 0-19 of each task (n_eval=20), at
    # the harness's warm-up, so every episode here pairs with one of theirs. VLA-Adapter
    # renders at 256 like its reference eval (the input it was trained on).
    "libero": {
        "checkpoint": "LIBERO-Long-Pro",
        "script": "flower/evaluation/flower_eval_libero.py",
        "suite_dir": "orig_libero_10",
        "member": "libero_orig.csv",
        "overrides": [
            "benchmark_name=libero_10",
            "dataset_path=/workspace",
            "log_dir=/saves/eval_logs",
            "env_img_res=256",
            "n_eval=20",
            "num_videos=0",
            "log_wandb=False",
        ],
    },
    "calvin": {
        "checkpoint": "CALVIN-ABC-Pro",
        "script": "flower/evaluation/flower_evaluate.py",
        "suite_dir": "calvin_ABCD_D",
        "member": "calvin.csv",
        "overrides": [],
    },
}


def checkpoint_dir(benchmark: str, ckpt_root: str) -> Path:
    return Path(ckpt_root) / BENCHMARKS[benchmark]["checkpoint"]


def result_csv(benchmark: str, ckpt_root: str) -> Path:
    """eval_records.result_dir(...)/result.csv for the unmasked evals of this checkpoint."""
    ckpt = checkpoint_dir(benchmark, ckpt_root)
    return ckpt / "eval_logs" / ckpt.name / BENCHMARKS[benchmark]["suite_dir"] / "result.csv"


def plan_lines(benchmark: str, ckpt_root: str, resume: bool) -> List[Tuple[str, List[str]]]:
    """All inputs, then each input removed, then each input masked."""
    spec = BENCHMARKS[benchmark]
    ckpt = checkpoint_dir(benchmark, ckpt_root)
    csv_path = result_csv(benchmark, ckpt_root)
    base = ["policy=vla_adapter", f"train_folder={ckpt}", f"checkpoint={ckpt}"] + spec["overrides"]
    combos = [FULL] + [{**FULL, key: False} for key in FULL]
    lines = [
        (spec["script"], base + modality_overrides(combo))
        for combo in combos
        if not (resume and combo_in_csv(csv_path, combo, use_proprio=True))
    ]
    return lines + masked_lines(spec["script"], base + modality_overrides(FULL), FULL, csv_path, True, resume)


def collect(benchmark: str, ckpt_root: str) -> Dict[str, Path]:
    """Copy the result.csv files to <ckpt>/artifact/<member>; returns member -> copy."""
    spec = BENCHMARKS[benchmark]
    csv_path = result_csv(benchmark, ckpt_root)
    out_dir = checkpoint_dir(benchmark, ckpt_root) / "artifact"
    sources = [(csv_path, spec["member"])] if csv_path.exists() else []
    sources += masked_artifact_members(csv_path, spec["member"])
    copies = {}
    for path, member in sources:
        target = out_dir / member
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        copies[member] = target
    return copies


def main(argv: List[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["plan", "collect"])
    parser.add_argument("--benchmark", choices=list(BENCHMARKS), required=True)
    parser.add_argument("--ckpt-root", default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--resume", action="store_true", help="skip evals result.csv already holds")
    args = parser.parse_args(argv)
    if args.command == "plan":
        for script, overrides in plan_lines(args.benchmark, args.ckpt_root, args.resume):
            print("\t".join([script] + overrides))
    else:
        for member, target in collect(args.benchmark, args.ckpt_root).items():
            print(f"{member} -> {target}")


if __name__ == "__main__":
    main(sys.argv[1:])
