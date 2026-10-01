#!/usr/bin/env python3
"""Decide, and later collect, the MimicGen evaluation a training run needs.

Sibling of scripts/eval_pipeline.py (the LIBERO/LIBERO-Plus planner), much smaller
because the first MimicGen run has one suite only: all-modalities-on evaluation across
every dataset in flower.datasets.mimicgen_tasks.CORE_DATASETS -- no dropout combos, no
LIBERO-Plus, no modality-off variants. Driven by ./run.sh pipeline-mimicgen.

Usage:
  python scripts/mimicgen_pipeline.py plan --train-folder <dir> [--reeval] [-- overrides...]
  python scripts/mimicgen_pipeline.py upload --train-folder <dir> [-- overrides...]

`plan` prints one line, tab-separated:
    eval-mimicgen\t<override1>\t<override2>\t...
where the overrides are Hydra "key=value" tokens for
flower/evaluation/flower_eval_mimicgen.py.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import wandb
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))
from eval_pipeline import artifact_name, parse_overrides, wandb_run_id  # noqa: E402

from flower.evaluation.eval_records import result_dir  # noqa: E402

LIBERO_VARIANT = "mimicgen"
SUITE = "core"


def _resolve(train_folder: str, extra_overrides: List[str]) -> Tuple[str, str]:
    """(resolved_train_folder, checkpoint), applying any user overrides.

    Mirrors eval_pipeline.py's _resolve, minus the LIBERO benchmark_name lookup --
    <train_folder>/.hydra/config.yaml for a MimicGen run has no `libero_benchmark` key.
    """
    train_cfg = OmegaConf.load(Path(train_folder) / ".hydra" / "config.yaml")
    parsed = parse_overrides(extra_overrides)
    resolved_train_folder = parsed.get("train_folder", str(train_folder))
    default_checkpoint = str(
        Path(train_folder) / f"seed_{train_cfg.seed}" / "saved_models" / "last.ckpt"
    )
    checkpoint = parsed.get("checkpoint", default_checkpoint)
    return resolved_train_folder, checkpoint


def _eval_overrides(train_folder: str, checkpoint: str) -> List[str]:
    return [
        f"train_folder={train_folder}",
        f"checkpoint={checkpoint}",
        "log_dir=/saves/eval_logs",
        "num_videos=0",
    ]


def plan_lines(
    train_folder: str, extra_overrides: List[str], reeval: bool
) -> List[Tuple[str, List[str]]]:
    resolved_train_folder, checkpoint = _resolve(train_folder, extra_overrides)
    overrides = _eval_overrides(resolved_train_folder, checkpoint)
    # Forward the rest (n_eval=..., eval_batch_size=...); train_folder/checkpoint are
    # already in `overrides`, resolved above.
    overrides += [
        o for o in extra_overrides if o.split("=", 1)[0] not in ("train_folder", "checkpoint")
    ]
    if reeval:
        overrides = overrides + ["reeval=True"]
    return [("eval-mimicgen", overrides)]


def upload(train_folder: str, extra_overrides: List[str]) -> None:
    train_cfg = OmegaConf.load(Path(train_folder) / ".hydra" / "config.yaml")
    resolved_train_folder, checkpoint = _resolve(train_folder, extra_overrides)
    csv_path = result_dir(resolved_train_folder, checkpoint, LIBERO_VARIANT, SUITE) / "result.csv"

    if not csv_path.exists():
        print(f"Nothing to upload: no result.csv found at {csv_path}.")
        return

    # Same run id as scripts/eval_pipeline.py's upload -- lands on the same W&B training
    # run, as a second "evaluation"-type artifact (LIBERO's own eval-<run_id> artifact,
    # if this run also went through ./run.sh pipeline, is untouched).
    run_id = wandb_run_id(resolved_train_folder)
    run = wandb.init(
        project=str(train_cfg.logger.project),
        entity=str(train_cfg.logger.entity),
        id=run_id,
        resume="allow",
    )
    artifact = wandb.Artifact(artifact_name(run_id), type="evaluation")
    artifact.add_file(str(csv_path), name="mimicgen.csv")
    run.log_artifact(artifact)
    run.finish()
    print(f"Logged eval-{run_id} artifact to {train_cfg.logger.project}/{run_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    plan_p = subparsers.add_parser("plan")
    plan_p.add_argument("--train-folder", required=True)
    plan_p.add_argument("--reeval", action="store_true")
    plan_p.add_argument("overrides", nargs="*")

    upload_p = subparsers.add_parser("upload")
    upload_p.add_argument("--train-folder", required=True)
    upload_p.add_argument("overrides", nargs="*")

    args = parser.parse_args()

    if args.cmd == "plan":
        for service, overrides in plan_lines(args.train_folder, args.overrides, args.reeval):
            print("\t".join([service] + overrides))
    elif args.cmd == "upload":
        upload(args.train_folder, args.overrides)


if __name__ == "__main__":
    main()
