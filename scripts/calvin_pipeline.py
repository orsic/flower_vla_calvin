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
  python scripts/calvin_pipeline.py upload-open-loop --train-folder <dir> [-- overrides...]
  python scripts/calvin_pipeline.py restore-open-loop --train-folder <dir> [-- overrides...]

`plan` prints one line per eval, tab-separated:
    eval-calvin\t<override1>\t<override2>\t...
where the overrides are Hydra "key=value" tokens for flower/evaluation/flower_evaluate.py.

`upload` attaches result.csv as calvin.csv and, when it holds more than one modality
combo, scripts/pid_modality.py's report as pid_modality.txt.

`upload-open-loop` attaches scripts/open_loop_calvin.py's parts
(<run>/open_loop/<ckpt>/parts/*.csv.gz) to their own artifact, open-loop-<run id>, as
<ckpt>/parts/<file>; `restore-open-loop` puts the missing ones back.
"""

import argparse
import re
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import wandb
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))
from eval_pipeline import (  # noqa: E402
    artifact_name,
    combo_in_csv,
    download_eval_artifact,
    masked_artifact_members,
    masked_lines,
    masked_restore_targets,
    modality_overrides,
    planned_combos,
    restore_files,
    trained_combo,
    wandb_run_id,
)
from mimicgen_pipeline import _eval_overrides, _resolve, write_pid_report  # noqa: E402

from flower.evaluation.eval_records import checkpoint_name, result_dir  # noqa: E402

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
    masked: bool = False,
) -> List[Tuple[str, List[str]]]:
    train_cfg = OmegaConf.load(Path(train_folder) / ".hydra" / "config.yaml")
    resolved_train_folder, checkpoint = _resolve(train_folder, extra_overrides)
    overrides = _eval_overrides(resolved_train_folder, checkpoint)
    # Forward the rest (num_sequences=..., ...); train_folder/checkpoint are already in
    # `overrides`, resolved above.
    overrides += [o for o in extra_overrides if o.split("=", 1)[0] not in ("train_folder", "checkpoint")]
    if masked:
        full = trained_combo(train_cfg)
        use_proprio = bool(train_cfg.model.get("use_proprio", False))
        path = csv_path(train_folder, extra_overrides)
        return masked_lines(SERVICE, overrides + modality_overrides(full), full, path, use_proprio, resume)
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
    for masked_path, name in masked_artifact_members(path, "calvin.csv"):
        artifact.add_file(str(masked_path), name=name)
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
        path = csv_path(train_folder, extra_overrides)
        targets = {"calvin.csv": path, **masked_restore_targets(path, "calvin.csv")}
        for path in restore_files(artifact_dir, targets):
            print(f"restore: {path}")


def open_loop_dir(train_folder: str, extra_overrides: List[str]) -> Path:
    resolved_train_folder, checkpoint = _resolve(train_folder, extra_overrides)
    return Path(resolved_train_folder) / "open_loop" / checkpoint_name(checkpoint)


def open_loop_artifact_name(run_id: str) -> str:
    return "open-loop-" + re.sub(r"[^A-Za-z0-9._-]", "-", run_id)


def upload_open_loop(train_folder: str, extra_overrides: List[str]) -> None:
    """Log every open-loop part as open-loop-<run id>. Each upload replaces the artifact's
    member list, so a worker restores the earlier parts first (./run.sh pipeline-calvin)."""
    out_dir = open_loop_dir(train_folder, extra_overrides)
    parts = sorted((out_dir / "parts").glob("*.csv.gz"))
    if not parts:
        print(f"Nothing to upload: no open-loop parts in {out_dir / 'parts'}.")
        return
    train_cfg = OmegaConf.load(Path(train_folder) / ".hydra" / "config.yaml")
    run_id = wandb_run_id(_resolve(train_folder, extra_overrides)[0])
    run = wandb.init(
        project=str(train_cfg.logger.project),
        entity=str(train_cfg.logger.entity),
        id=run_id,
        resume="allow",
    )
    artifact = wandb.Artifact(open_loop_artifact_name(run_id), type="open_loop")
    for part in parts:
        artifact.add_file(str(part), name=f"{out_dir.name}/parts/{part.name}")
    run.log_artifact(artifact)
    run.finish()
    print(f"Logged {open_loop_artifact_name(run_id)} ({len(parts)} parts) to {train_cfg.logger.project}/{run_id}")


def download_open_loop_artifact(train_cfg, run_id: str, root: Path) -> Optional[Path]:
    """Download the run's latest open-loop artifact into root; None if it has none."""
    name = f"{train_cfg.logger.entity}/{train_cfg.logger.project}/{open_loop_artifact_name(run_id)}:latest"
    try:
        artifact = wandb.Api(timeout=120).artifact(name, type="open_loop")
    except wandb.errors.CommError:
        return None
    return Path(artifact.download(root=str(root)))


def restore_open_loop(train_folder: str, extra_overrides: List[str]) -> None:
    """Put the open-loop parts of the run's W&B artifact that are missing locally back."""
    train_cfg = OmegaConf.load(Path(train_folder) / ".hydra" / "config.yaml")
    out_dir = open_loop_dir(train_folder, extra_overrides)
    run_id = wandb_run_id(_resolve(train_folder, extra_overrides)[0])
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = download_open_loop_artifact(train_cfg, run_id, Path(tmp))
        if artifact_dir is None:
            print("restore-open-loop: the run has no open-loop artifact -- nothing to restore")
            return
        members = (artifact_dir / out_dir.name / "parts").glob("*.csv.gz")
        targets = {f"{out_dir.name}/parts/{m.name}": out_dir / "parts" / m.name for m in members}
        for path in restore_files(artifact_dir, targets):
            print(f"restore-open-loop: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    plan_p = subparsers.add_parser("plan")
    plan_p.add_argument("--train-folder", required=True)
    plan_p.add_argument("--reeval", action="store_true")
    plan_p.add_argument("--resume", action="store_true")
    plan_p.add_argument("--all-combos", action="store_true")
    plan_p.add_argument("--skip-multi-off", action="store_true")
    plan_p.add_argument("--masked", action="store_true", help="only the masked-input evals (MASKED_VARIANTS)")
    plan_p.add_argument("overrides", nargs="*")

    upload_p = subparsers.add_parser("upload")
    upload_p.add_argument("--train-folder", required=True)
    upload_p.add_argument("overrides", nargs="*")

    restore_p = subparsers.add_parser("restore")
    restore_p.add_argument("--train-folder", required=True)
    restore_p.add_argument("overrides", nargs="*")

    for name in ("upload-open-loop", "restore-open-loop"):
        open_loop_p = subparsers.add_parser(name)
        open_loop_p.add_argument("--train-folder", required=True)
        open_loop_p.add_argument("overrides", nargs="*")

    args = parser.parse_args()

    if args.cmd == "restore":
        restore(args.train_folder, args.overrides)
    elif args.cmd == "plan":
        for service, overrides in plan_lines(
            args.train_folder,
            args.overrides,
            args.reeval,
            args.all_combos,
            args.skip_multi_off,
            args.resume,
            args.masked,
        ):
            print("\t".join([service] + overrides))
    elif args.cmd == "upload":
        upload(args.train_folder, args.overrides)
    elif args.cmd == "upload-open-loop":
        upload_open_loop(args.train_folder, args.overrides)
    elif args.cmd == "restore-open-loop":
        restore_open_loop(args.train_folder, args.overrides)


if __name__ == "__main__":
    main()
