#!/usr/bin/env python3
"""Decide, and later collect, the MimicGen evaluation a training run needs.

Sibling of scripts/eval_pipeline.py (the LIBERO/LIBERO-Plus planner), much smaller
because MimicGen has one suite only (every dataset in
flower.datasets.mimicgen_tasks.CORE_DATASETS) -- no LIBERO-Plus, no modality-off
variants. A plain run gets one all-modalities eval; a model.modality_dropout=True run
gets one eval per modality combo (eval_pipeline.modality_combos: 7, or 14 with
model.use_proprio=True), all merging into the same result.csv. Driven by
./run.sh pipeline-mimicgen. --all-combos sweeps a plain run the same way (within its
trained inputs), --skip-multi-off keeps the trained inputs and each one withheld, and
--resume skips the combos result.csv already holds.

Usage:
  python scripts/mimicgen_pipeline.py plan --train-folder <dir> [--reeval] [--resume]
      [--all-combos] [--skip-multi-off] [-- overrides...]
  python scripts/mimicgen_pipeline.py upload --train-folder <dir> [-- overrides...]
  python scripts/mimicgen_pipeline.py restore --train-folder <dir> [-- overrides...]

`plan` prints one line per eval, tab-separated:
    eval-mimicgen\t<override1>\t<override2>\t...
where the overrides are Hydra "key=value" tokens for
flower/evaluation/flower_eval_mimicgen.py.

`upload` attaches result.csv as mimicgen.csv and, when it holds more than one modality
combo, scripts/pid_modality.py's report as pid_modality.txt.
"""

import argparse
import subprocess
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
    modality_overrides,
    parse_overrides,
    planned_combos,
    restore_files,
    wandb_run_id,
)
from perturbation_sr import modality_combos_present  # noqa: E402
from pid_modality import load_rows  # noqa: E402

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


def csv_path(train_folder: str, extra_overrides: List[str]) -> Path:
    resolved_train_folder, checkpoint = _resolve(train_folder, extra_overrides)
    return result_dir(resolved_train_folder, checkpoint, LIBERO_VARIANT, SUITE) / "result.csv"


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
    # Forward the rest (n_eval=..., eval_batch_size=...); train_folder/checkpoint are
    # already in `overrides`, resolved above.
    overrides += [
        o for o in extra_overrides if o.split("=", 1)[0] not in ("train_folder", "checkpoint")
    ]
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
        # The combo's eval_modalities.* come after the forwarded overrides (Hydra keeps
        # the last occurrence), so the sweep wins. reeval only on the first line: every
        # line merges into the same result.csv, and reeval on a later one would rotate
        # away the combos the earlier lines just wrote.
        line = overrides + (modality_overrides(combo) if named else [])
        if reeval and i == 0:
            line.append("reeval=True")
        lines.append(("eval-mimicgen", line))
    return lines


def write_pid_report(csv_path: Path) -> Optional[Path]:
    """Run scripts/pid_modality.py on a multi-combo result.csv; its report is written next
    to it as pid_modality.txt. None for a single-combo CSV (nothing to decompose) or if
    pid_modality.py fails -- that must not cost the upload, as in eval_pipeline.upload."""
    if len(modality_combos_present(load_rows(str(csv_path)))) < 2:
        return None
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "pid_modality.py"), str(csv_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(
            f"WARNING: pid_modality.py failed on {csv_path} ({exc}; stderr: {exc.stderr}) "
            "-- skipping pid_modality.txt",
            file=sys.stderr,
        )
        return None
    pid_txt = csv_path.parent / "pid_modality.txt"
    pid_txt.write_text(result.stdout)
    print(result.stdout)
    return pid_txt


def upload(train_folder: str, extra_overrides: List[str]) -> None:
    train_cfg = OmegaConf.load(Path(train_folder) / ".hydra" / "config.yaml")
    resolved_train_folder, _ = _resolve(train_folder, extra_overrides)
    path = csv_path(train_folder, extra_overrides)

    if not path.exists():
        print(f"Nothing to upload: no result.csv found at {path}.")
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
    artifact.add_file(str(path), name="mimicgen.csv")
    pid_txt = write_pid_report(path)
    if pid_txt is not None:
        artifact.add_file(str(pid_txt), name="pid_modality.txt")
    run.log_artifact(artifact)
    run.finish()
    print(f"Logged eval-{run_id} artifact to {train_cfg.logger.project}/{run_id}")


def restore(train_folder: str, extra_overrides: List[str]) -> None:
    """Put mimicgen.csv of the run's latest W&B evaluation artifact back as result.csv unless one
    exists locally (PIPELINE_RESTORE=1; see eval_pipeline.restore)."""
    train_cfg = OmegaConf.load(Path(train_folder) / ".hydra" / "config.yaml")
    resolved_train_folder, _ = _resolve(train_folder, extra_overrides)
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = download_eval_artifact(train_cfg, wandb_run_id(resolved_train_folder), Path(tmp))
        if artifact_dir is None:
            print("restore: the run has no evaluation artifact -- nothing to restore")
            return
        for path in restore_files(artifact_dir, {"mimicgen.csv": csv_path(train_folder, extra_overrides)}):
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
