#!/usr/bin/env python3
"""Decide, and later collect, the LIBERO/LIBERO-Plus evaluations a training run needs.

The heavy lifting (running one eval, writing/merging result.csv) stays in
flower/evaluation/flower_eval_libero.py; this script only figures out *which* evals to run
for a given training run and gathers their output afterwards. Driven by ./run.sh pipeline.

Usage:
  python scripts/eval_pipeline.py plan --train-folder <dir> [--resume] [-- overrides...]
  python scripts/eval_pipeline.py upload --train-folder <dir> [-- overrides...]

`plan` prints one line per evaluation to run, tab-separated:
    <service>\t<override1>\t<override2>\t...
where <service> is "eval" or "eval-plus" (a run.sh podman-compose service name) and the
overrides are Hydra "key=value" tokens for flower/evaluation/flower_eval_libero.py.

Branching (read from <train-folder>/.hydra/config.yaml):
  - model.modality_dropout=False: one full-modality "eval" line, one full-modality
    "eval-plus" line.
  - model.modality_dropout=True: one "eval" line per non-empty combination of
    (rgb_static, rgb_gripper, language) -- 7 combos -- crossed with proprio on/off when
    model.use_proprio=True (14 combos total), plus one full-modality "eval-plus" line.
    "proprio only" is never emitted: flower_eval_libero.py rejects an eval with all three
    token modalities off.

--overrides (after a literal "--") are Hydra overrides appended, unmodified, to every
emitted line -- last on the line, so they take precedence (see run.sh's run_train/eval
helpers for the same last-wins convention). They are also inspected for the handful of
keys (benchmark_name, checkpoint, train_folder, csv_dir) that change which result.csv
`plan --resume` and `upload` read, so overriding e.g. benchmark_name redirects both. Do not
pass eval_modalities.* through -- it would fight the combo sweep.

`upload` runs scripts/pid_modality.py on the LIBERO (orig) result.csv, then attaches that
output plus both result.csv files to a W&B artifact on the *training* run (same project/
entity/id as training_libero.py's setup_logger derives from the run dir).
"""
import argparse
import itertools
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import wandb
from omegaconf import OmegaConf

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())
from flower.evaluation.eval_records import checkpoint_name, read_csv  # noqa: E402

TOKEN_MODALITIES = ["rgb_static", "rgb_gripper", "language"]
FLAG_COLUMNS = {
    "rgb_static": "use_rgb_static",
    "rgb_gripper": "use_rgb_gripper",
    "language": "use_language",
    "proprio": "use_proprio",
}


def full_modality_combo() -> Dict[str, bool]:
    return {"rgb_static": True, "rgb_gripper": True, "language": True, "proprio": True}


def token_combos() -> List[Dict[str, bool]]:
    """All non-empty combinations of the three token modalities, all-on first, then by
    descending count of enabled modalities."""
    combos = []
    for r in range(len(TOKEN_MODALITIES), 0, -1):
        for on in itertools.combinations(TOKEN_MODALITIES, r):
            combos.append({m: (m in on) for m in TOKEN_MODALITIES})
    return combos


def modality_combos(use_proprio: bool) -> List[Dict[str, bool]]:
    """7 combos (proprio always on) if the model never receives proprio, else 14 (each
    token combo x proprio on/off). "proprio only" is never a token combo here, so it's
    never crossed either -- matching flower_eval_libero.py's all-tokens-off rejection."""
    combos = []
    for token_combo in token_combos():
        if use_proprio:
            for proprio in (True, False):
                combos.append({**token_combo, "proprio": proprio})
        else:
            combos.append({**token_combo, "proprio": True})
    return combos


def parse_overrides(overrides: List[str]) -> Dict[str, str]:
    """"key=value" tokens -> dict, last occurrence wins (mirrors Hydra's own semantics)."""
    parsed = {}
    for override in overrides:
        key, sep, value = override.partition("=")
        if sep:
            parsed[key] = value
    return parsed


def _csv_path(train_folder: str, checkpoint: str, variant: str, benchmark_name: str) -> Path:
    """Same layout as eval_records.result_dir, without its mkdir side effect (plan/upload
    only read this path, never create it)."""
    return (
        Path(train_folder)
        / "eval_logs"
        / checkpoint_name(checkpoint)
        / f"{variant}_{benchmark_name}"
        / "result.csv"
    )


def _matches(row: Dict[str, str], combo: Dict[str, bool]) -> bool:
    return all(int(row.get(col) or 0) == int(combo[key]) for key, col in FLAG_COLUMNS.items())


def _eval_overrides(benchmark_name: str, train_folder: str, checkpoint: str, combo: Dict[str, bool]) -> List[str]:
    overrides = [
        f"benchmark_name={benchmark_name}",
        f"train_folder={train_folder}",
        f"checkpoint={checkpoint}",
        "dataset_path=/workspace",
        "log_dir=/saves/eval_logs",
        "num_videos=0",
        "log_wandb=False",
    ]
    for key in ("rgb_static", "rgb_gripper", "language", "proprio"):
        overrides.append(f"eval_modalities.{key}={combo[key]}")
    return overrides


def _resolve(train_folder: str, extra_overrides: List[str]) -> Tuple[str, str, str]:
    """(benchmark_name, resolved_train_folder, checkpoint), applying any user overrides."""
    train_cfg = OmegaConf.load(Path(train_folder) / ".hydra" / "config.yaml")
    parsed = parse_overrides(extra_overrides)
    resolved_train_folder = parsed.get("train_folder", str(train_folder))
    benchmark_name = parsed.get("benchmark_name", str(train_cfg.libero_benchmark))
    default_checkpoint = str(
        Path(train_folder) / f"seed_{train_cfg.seed}" / "saved_models" / "last.ckpt"
    )
    checkpoint = parsed.get("checkpoint", default_checkpoint)
    return benchmark_name, resolved_train_folder, checkpoint


def plan_lines(train_folder: str, resume: bool, extra_overrides: List[str]) -> List[Tuple[str, List[str]]]:
    train_cfg = OmegaConf.load(Path(train_folder) / ".hydra" / "config.yaml")
    dropout = bool(train_cfg.model.modality_dropout)
    use_proprio = bool(train_cfg.model.use_proprio)
    benchmark_name, resolved_train_folder, checkpoint = _resolve(train_folder, extra_overrides)
    parsed = parse_overrides(extra_overrides)

    def already_done(variant: str, combo: Dict[str, bool]) -> bool:
        if not resume:
            return False
        csv_dir_override = parsed.get("csv_dir")
        csv_path = (
            Path(csv_dir_override) / "result.csv"
            if csv_dir_override
            else _csv_path(resolved_train_folder, checkpoint, variant, benchmark_name)
        )
        # A model with use_proprio=False never receives proprio regardless of
        # eval_modalities.proprio (flower_eval_libero.py's EvaluateLibero.uses_proprio),
        # so that's what actually lands in the CSV's use_proprio column -- match on it,
        # not on the raw (always-True) combo value, or a no-proprio run never resumes.
        effective_combo = {**combo, "proprio": combo["proprio"] and use_proprio}
        return any(_matches(row, effective_combo) for row in read_csv(csv_path))

    combos = modality_combos(use_proprio) if dropout else [full_modality_combo()]

    lines: List[Tuple[str, List[str]]] = []
    for combo in combos:
        if already_done("orig", combo):
            continue
        overrides = _eval_overrides(benchmark_name, resolved_train_folder, checkpoint, combo)
        lines.append(("eval", overrides + extra_overrides))

    plus_combo = full_modality_combo()
    if not already_done("plus", plus_combo):
        overrides = _eval_overrides(benchmark_name, resolved_train_folder, checkpoint, plus_combo)
        lines.append(("eval-plus", overrides + extra_overrides))

    return lines


def wandb_run_id(train_folder: str) -> str:
    """Reconstruct the deterministic id training_libero.py's setup_logger assigns:
    id = f"{run_dir.parent.name}_{run_dir.name}" (flower/training_libero.py:58-64)."""
    resolved = Path(train_folder).resolve()
    return f"{resolved.parent.name}_{resolved.name}"


def upload(train_folder: str, extra_overrides: List[str]) -> None:
    train_cfg = OmegaConf.load(Path(train_folder) / ".hydra" / "config.yaml")
    benchmark_name, resolved_train_folder, checkpoint = _resolve(train_folder, extra_overrides)

    orig_csv = _csv_path(resolved_train_folder, checkpoint, "orig", benchmark_name)
    plus_csv = _csv_path(resolved_train_folder, checkpoint, "plus", benchmark_name)

    pid_txt = None
    if orig_csv.exists():
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "pid_modality.py"), str(orig_csv)],
            capture_output=True,
            text=True,
            check=True,
        )
        pid_txt = orig_csv.parent / "pid_modality.txt"
        pid_txt.write_text(result.stdout)
        print(result.stdout)

    if not orig_csv.exists() and not plus_csv.exists():
        print(f"Nothing to upload: neither {orig_csv} nor {plus_csv} exists.")
        return

    run_id = wandb_run_id(resolved_train_folder)
    run = wandb.init(
        project=str(train_cfg.logger.project),
        entity=str(train_cfg.logger.entity),
        id=run_id,
        resume="allow",
    )
    artifact = wandb.Artifact(f"eval-{run_id}", type="evaluation")
    if orig_csv.exists():
        artifact.add_file(str(orig_csv), name="libero_orig.csv")
    if plus_csv.exists():
        artifact.add_file(str(plus_csv), name="libero_plus.csv")
    if pid_txt is not None:
        artifact.add_file(str(pid_txt), name="pid_modality.txt")
    run.log_artifact(artifact)
    run.finish()
    print(f"Logged eval-{run_id} artifact to {train_cfg.logger.project}/{run_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    plan_p = subparsers.add_parser("plan")
    plan_p.add_argument("--train-folder", required=True)
    plan_p.add_argument("--resume", action="store_true")
    plan_p.add_argument("overrides", nargs="*")

    upload_p = subparsers.add_parser("upload")
    upload_p.add_argument("--train-folder", required=True)
    upload_p.add_argument("overrides", nargs="*")

    args = parser.parse_args()

    if args.cmd == "plan":
        for service, overrides in plan_lines(args.train_folder, args.resume, args.overrides):
            print("\t".join([service] + overrides))
    elif args.cmd == "upload":
        upload(args.train_folder, args.overrides)


if __name__ == "__main__":
    main()
