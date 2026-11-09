#!/usr/bin/env python3
"""Decide, and later collect, the CALVIN evaluation a training run needs.

Sibling of scripts/mimicgen_pipeline.py (reusing its checkpoint resolution and PID
report): a plain run gets one all-modalities long-horizon eval; a
model.modality_dropout=True run gets one eval per modality combo
(eval_pipeline.modality_combos: 7, or 14 with model.use_proprio=True), all merging into
the same result.csv. Driven by ./run.sh pipeline-calvin. --all-combos sweeps a plain run
the same way (within its trained inputs), --skip-multi-off keeps the trained inputs and
each one withheld, and --resume skips the combos result.csv already holds.

Usage:
  python scripts/calvin_pipeline.py plan --train-folder <dir> [--reeval] [--resume]
      [--all-combos] [--skip-multi-off] [-- overrides...]
  python scripts/calvin_pipeline.py upload --train-folder <dir> [-- overrides...]
  python scripts/calvin_pipeline.py restore --train-folder <dir> [-- overrides...]

`plan` prints one line per eval, tab-separated:
    eval-calvin\t<override1>\t<override2>\t...
where the overrides are Hydra "key=value" tokens for flower/evaluation/flower_evaluate.py.

`upload` attaches result.csv as calvin.csv and, when it holds more than one modality
combo, scripts/pid_modality.py's report as pid_modality.txt.
"""

import argparse
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import wandb
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))
from eval_pipeline import (  # noqa: E402
    artifact_name,
    combo_in_csv,
    download_eval_artifact,
    modality_overrides,
    planned_combos,
    restore_files,
    wandb_run_id,
)
from mimicgen_pipeline import _eval_overrides, _resolve, write_pid_report  # noqa: E402

from flower.evaluation.eval_records import result_dir  # noqa: E402

LIBERO_VARIANT = "calvin"
SUITE = "ABCD_D"
SERVICE = "eval-calvin"


def plan_lines(
    train_folder: str,
    extra_overrides: List[str],
    reeval: bool,
    all_combos: bool = False,
    skip_multi_off: bool = False,
    resume: bool = False,
) -> List[Tuple[str, List[str]]]:
    train_cfg = OmegaConf.load(Path(train_folder) / ".hydra" / "config.yaml")
    resolved_train_folder, checkpoint = _resolve(train_folder, extra_overrides)
    overrides = _eval_overrides(resolved_train_folder, checkpoint)
    # Forward the rest (num_sequences=..., ...); train_folder/checkpoint are already in
    # `overrides`, resolved above.
    overrides += [o for o in extra_overrides if o.split("=", 1)[0] not in ("train_folder", "checkpoint")]
    combos = planned_combos(train_cfg, all_combos, skip_multi_off)
    # A plain run's single all-inputs line names no modalities; a sweep, or a fixed-subset
    # run's trained inputs, are named so result.csv's use_* columns record them.
    named = len(combos) > 1 or not all(combos[0].values()) or all_combos
    if resume:
        path = csv_path(train_folder, extra_overrides)
        use_proprio = bool(train_cfg.model.get("use_proprio", False))
        combos = [c for c in combos if not combo_in_csv(path, c, use_proprio)]

    lines = []
    for i, combo in enumerate(combos):
        # Same ordering rules as mimicgen_pipeline.plan_lines: the combo wins over forwarded
        # overrides, and reeval only on the first line (every line merges into one result.csv).
        line = overrides + (modality_overrides(combo) if named else [])
        if reeval and i == 0:
            line.append("reeval=True")
        lines.append((SERVICE, line))
    return lines


def csv_path(train_folder: str, extra_overrides: List[str]) -> Path:
    resolved_train_folder, checkpoint = _resolve(train_folder, extra_overrides)
    return result_dir(resolved_train_folder, checkpoint, LIBERO_VARIANT, SUITE) / "result.csv"


def upload(train_folder: str, extra_overrides: List[str]) -> None:
    train_cfg = OmegaConf.load(Path(train_folder) / ".hydra" / "config.yaml")
    resolved_train_folder, _ = _resolve(train_folder, extra_overrides)
    path = csv_path(train_folder, extra_overrides)
    if not path.exists():
        print(f"Nothing to upload: no result.csv found at {path}.")
        return

    # Same run id as the training logger -- lands on the CALVIN training run itself.
    run_id = wandb_run_id(resolved_train_folder)
    run = wandb.init(
        project=str(train_cfg.logger.project),
        entity=str(train_cfg.logger.entity),
        id=run_id,
        resume="allow",
    )
    artifact = wandb.Artifact(artifact_name(run_id), type="evaluation")
    artifact.add_file(str(path), name="calvin.csv")
    pid_txt = write_pid_report(path)
    if pid_txt is not None:
        artifact.add_file(str(pid_txt), name="pid_modality.txt")
    run.log_artifact(artifact)
    run.finish()
    print(f"Logged eval-{run_id} artifact to {train_cfg.logger.project}/{run_id}")


def restore(train_folder: str, extra_overrides: List[str]) -> None:
    """Put calvin.csv of the run's latest W&B evaluation artifact back as result.csv unless one
    exists locally (PIPELINE_RESTORE=1; see eval_pipeline.restore)."""
    train_cfg = OmegaConf.load(Path(train_folder) / ".hydra" / "config.yaml")
    resolved_train_folder, _ = _resolve(train_folder, extra_overrides)
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = download_eval_artifact(train_cfg, wandb_run_id(resolved_train_folder), Path(tmp))
        if artifact_dir is None:
            print("restore: the run has no evaluation artifact -- nothing to restore")
            return
        for path in restore_files(artifact_dir, {"calvin.csv": csv_path(train_folder, extra_overrides)}):
            print(f"restore: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    plan_p = subparsers.add_parser("plan")
    plan_p.add_argument("--train-folder", required=True)
    plan_p.add_argument("--reeval", action="store_true")
    plan_p.add_argument("--resume", action="store_true")
    plan_p.add_argument("--all-combos", action="store_true")
    plan_p.add_argument("--skip-multi-off", action="store_true")
    plan_p.add_argument("overrides", nargs="*")

    upload_p = subparsers.add_parser("upload")
    upload_p.add_argument("--train-folder", required=True)
    upload_p.add_argument("overrides", nargs="*")

    restore_p = subparsers.add_parser("restore")
    restore_p.add_argument("--train-folder", required=True)
    restore_p.add_argument("overrides", nargs="*")

    args = parser.parse_args()

    if args.cmd == "restore":
        restore(args.train_folder, args.overrides)
    elif args.cmd == "plan":
        for service, overrides in plan_lines(
            args.train_folder, args.overrides, args.reeval, args.all_combos, args.skip_multi_off, args.resume
        ):
            print("\t".join([service] + overrides))
    elif args.cmd == "upload":
        upload(args.train_folder, args.overrides)


if __name__ == "__main__":
    main()
