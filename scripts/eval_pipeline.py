#!/usr/bin/env python3
"""Decide, and later collect, the LIBERO/LIBERO-Plus evaluations a training run needs.

The heavy lifting (running one eval, writing/merging result.csv) stays in
flower/evaluation/flower_eval_libero.py; this script only figures out *which* evals to run
for a given training run and gathers their output afterwards. Driven by ./run.sh pipeline.

Usage:
  python scripts/eval_pipeline.py plan --train-folder <dir> [--resume]
      [--reeval [--reeval-suites orig_<bench>,plus_<bench>]] [-- overrides...]
  python scripts/eval_pipeline.py upload --train-folder <dir> [-- overrides...]

`plan` prints one line per evaluation to run, tab-separated:
    <service>\t<override1>\t<override2>\t...
where <service> is "eval" or "eval-plus" (a run.sh podman-compose service name) and the
overrides are Hydra "key=value" tokens for flower/evaluation/flower_eval_libero.py.

`--reeval` (driven by run.sh pipeline's PIPELINE_REEVAL=1) backs the selected suites'
result.csv up to results_<mtime>.csv (named for the old file's own mtime) and plans only
those suites, instead of the default merge-into-existing behavior. `--reeval-suites`
(PIPELINE_REEVAL_SUITES) narrows which suites -- a comma-separated list of result
directory names (e.g. "plus_libero_10"); omitted means every suite for this benchmark.
Rotation happens here, once, before planning -- never per eval invocation, since a
dropout run's several `eval` lines all merge into the same orig_<bench>/result.csv, and
never after planning, since --resume reading the not-yet-rotated file would mark
everything done and hand back an empty plan.

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

`upload` runs scripts/pid_modality.py on the LIBERO (orig) result.csv and
scripts/severity_sr.py on the LIBERO-Plus result.csv, then attaches that output plus both
result.csv files to a W&B artifact on the *training* run (same project/entity/id as
training_libero.py's setup_logger derives from the run dir). Since `plan --resume` treats
every combo already in a result.csv as done, `PIPELINE_RESUME=1 ./run.sh pipeline
<train_run_dir>` on a fully-evaluated run plans nothing and goes straight to `upload` --
the way to regenerate and re-upload these derived files without re-evaluating anything.
"""
import argparse
import itertools
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import wandb
from omegaconf import OmegaConf

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())
from flower.evaluation.eval_records import checkpoint_name, read_csv, rotate_result_csv  # noqa: E402

import severity_sr  # noqa: E402 -- same-directory import, python puts scripts/ on sys.path[0]

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


def _variant_csv_path(
    variant: str,
    parsed: Dict[str, str],
    resolved_train_folder: str,
    checkpoint: str,
    benchmark_name: str,
) -> Path:
    """result.csv path for one variant, honoring a csv_dir= override the same way
    already_done and rotate_reeval_csvs must -- kept in one place so the two can't drift
    into rotating a file the eval isn't actually going to write."""
    csv_dir_override = parsed.get("csv_dir")
    if csv_dir_override:
        return Path(csv_dir_override) / "result.csv"
    return _csv_path(resolved_train_folder, checkpoint, variant, benchmark_name)


def resolve_reeval_variants(suites_arg: Optional[str], benchmark_name: str) -> Set[str]:
    """Parse --reeval-suites into a set of variants ("orig"/"plus") to re-evaluate.

    suites_arg is None when --reeval-suites wasn't passed at all -- re-evaluate every
    suite for this benchmark. When passed, each comma-separated token must name a result
    directory exactly (e.g. "plus_libero_10"), checked against the *resolved*
    benchmark_name so a trailing benchmark_name= override is honored. A token for a
    different benchmark is almost certainly a mistake, not something to silently ignore.
    """
    valid = {f"orig_{benchmark_name}": "orig", f"plus_{benchmark_name}": "plus"}
    if suites_arg is None:
        return set(valid.values())
    tokens = [token.strip() for token in suites_arg.split(",") if token.strip()]
    if not tokens:
        raise ValueError(
            f"--reeval-suites was given but named no suite -- expected a comma-separated "
            f"list from {sorted(valid)}"
        )
    variants: Set[str] = set()
    for token in tokens:
        if token not in valid:
            raise ValueError(
                f"--reeval-suites: {token!r} is not a result directory for benchmark "
                f"{benchmark_name!r} -- expected one of {sorted(valid)}"
            )
        variants.add(valid[token])
    return variants


def rotate_reeval_csvs(
    variants: Set[str],
    parsed: Dict[str, str],
    resolved_train_folder: str,
    checkpoint: str,
    benchmark_name: str,
) -> List[Optional[Path]]:
    """Back up each selected variant's result.csv so plan_lines starts it from empty.

    Must be called once, before plan_lines -- see the module docstring for why rotating
    per-invocation or after planning both silently lose data.
    """
    csv_dir_override = parsed.get("csv_dir")
    if csv_dir_override and variants != {"orig", "plus"}:
        raise ValueError(
            f"csv_dir={csv_dir_override} makes orig and plus share one result.csv; "
            "--reeval-suites cannot rotate one without discarding the other. Select "
            "both suites, or drop csv_dir."
        )

    paths: List[Path] = []
    seen: Set[Path] = set()
    for variant in sorted(variants):
        path = _variant_csv_path(variant, parsed, resolved_train_folder, checkpoint, benchmark_name)
        if path in seen:
            continue  # csv_dir= makes both variants share one path -- rotate it once
        seen.add(path)
        paths.append(path)

    backups: List[Optional[Path]] = []
    for path in paths:
        backup = rotate_result_csv(path)
        if backup is not None:
            print(f"reeval: rotated {path} -> {backup}", file=sys.stderr)
        else:
            print(f"reeval: nothing to rotate at {path}", file=sys.stderr)
        backups.append(backup)
    return backups


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


def plan_lines(
    train_folder: str,
    resume: bool,
    extra_overrides: List[str],
    reeval_variants: Optional[Set[str]] = None,
) -> List[Tuple[str, List[str]]]:
    """reeval_variants, when given, restricts the plan to those variants only -- the
    caller (main()) is expected to have already rotated their result.csv files aside, so
    `resume`'s "already done" check naturally finds nothing and replans them in full."""
    train_cfg = OmegaConf.load(Path(train_folder) / ".hydra" / "config.yaml")
    dropout = bool(train_cfg.model.modality_dropout)
    use_proprio = bool(train_cfg.model.use_proprio)
    benchmark_name, resolved_train_folder, checkpoint = _resolve(train_folder, extra_overrides)
    parsed = parse_overrides(extra_overrides)

    def already_done(variant: str, combo: Dict[str, bool]) -> bool:
        if not resume:
            return False
        csv_path = _variant_csv_path(variant, parsed, resolved_train_folder, checkpoint, benchmark_name)
        # A model with use_proprio=False never receives proprio regardless of
        # eval_modalities.proprio (flower_eval_libero.py's EvaluateLibero.uses_proprio),
        # so that's what actually lands in the CSV's use_proprio column -- match on it,
        # not on the raw (always-True) combo value, or a no-proprio run never resumes.
        effective_combo = {**combo, "proprio": combo["proprio"] and use_proprio}
        return any(_matches(row, effective_combo) for row in read_csv(csv_path))

    combos = modality_combos(use_proprio) if dropout else [full_modality_combo()]

    lines: List[Tuple[str, List[str]]] = []
    if reeval_variants is None or "orig" in reeval_variants:
        for combo in combos:
            if already_done("orig", combo):
                continue
            overrides = _eval_overrides(benchmark_name, resolved_train_folder, checkpoint, combo)
            lines.append(("eval", overrides + extra_overrides))

    if reeval_variants is None or "plus" in reeval_variants:
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


def artifact_name(run_id: str) -> str:
    """W&B artifact names allow only [A-Za-z0-9._-]; a modality-ablation run id carries
    '+' from run.sh's modality_label (e.g. "libero_10_static+wrist_...", run.sh:77). The
    W&B *run* id (wandb.init) keeps the '+' -- only the artifact name is sanitized."""
    return "eval-" + re.sub(r"[^A-Za-z0-9._-]", "-", run_id)


def write_severity_csv(plus_csv: Path, orig_csv: Optional[Path] = None) -> Optional[Path]:
    """severity_sr.csv next to plus_csv, or None if plus_csv is absent or the
    severity computation fails (e.g. LIBERO-Plus assets not downloaded -- Light
    Conditions' light_severity reads scene XMLs from there). A missing/broken asset
    checkout must not cost the whole artifact upload, so this warns and returns None
    rather than raising. `orig_csv`, when given and present, adds the init-state-matched
    LIBERO original baseline columns (see severity_sr.py's module docstring); absent,
    the baseline columns are left empty, same as calling severity_sr.py with no
    --orig-csv."""
    if not plus_csv.exists():
        return None
    orig_rows = severity_sr.load_rows(str(orig_csv)) if orig_csv is not None and orig_csv.exists() else None
    try:
        records = severity_sr.collect(
            severity_sr.load_rows(str(plus_csv)), severity_sr.DEFAULT_LIBERO_PLUS_ROOT, orig_rows=orig_rows
        )
    except Exception as exc:  # noqa: BLE001 -- see docstring
        print(f"WARNING: severity_sr.py failed on {plus_csv} ({exc}) -- skipping severity_sr.csv", file=sys.stderr)
        return None
    csv_path = plus_csv.parent / "severity_sr.csv"
    severity_sr.write_csv(csv_path, records)
    return csv_path


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

    severity_csv = write_severity_csv(plus_csv, orig_csv)

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
    artifact = wandb.Artifact(artifact_name(run_id), type="evaluation")
    if orig_csv.exists():
        artifact.add_file(str(orig_csv), name="libero_orig.csv")
    if plus_csv.exists():
        artifact.add_file(str(plus_csv), name="libero_plus.csv")
    if pid_txt is not None:
        artifact.add_file(str(pid_txt), name="pid_modality.txt")
    if severity_csv is not None:
        artifact.add_file(str(severity_csv), name="severity_sr.csv")
    run.log_artifact(artifact)
    run.finish()
    print(f"Logged eval-{run_id} artifact to {train_cfg.logger.project}/{run_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    plan_p = subparsers.add_parser("plan")
    plan_p.add_argument("--train-folder", required=True)
    plan_p.add_argument("--resume", action="store_true")
    plan_p.add_argument("--reeval", action="store_true")
    plan_p.add_argument("--reeval-suites", default=None)
    plan_p.add_argument("overrides", nargs="*")

    upload_p = subparsers.add_parser("upload")
    upload_p.add_argument("--train-folder", required=True)
    upload_p.add_argument("overrides", nargs="*")

    args = parser.parse_args()

    if args.cmd == "plan":
        if args.reeval_suites is not None and not args.reeval:
            sys.exit("--reeval-suites requires --reeval")

        reeval_variants: Optional[Set[str]] = None
        if args.reeval:
            benchmark_name, resolved_train_folder, checkpoint = _resolve(args.train_folder, args.overrides)
            parsed = parse_overrides(args.overrides)
            try:
                reeval_variants = resolve_reeval_variants(args.reeval_suites, benchmark_name)
                # Rotation must happen here, before plan_lines: see the module docstring
                # and plan_lines' own docstring for why the reverse order silently loses
                # data (resume would read the not-yet-rotated file and plan nothing).
                rotate_reeval_csvs(reeval_variants, parsed, resolved_train_folder, checkpoint, benchmark_name)
            except ValueError as exc:
                sys.exit(str(exc))

        for service, overrides in plan_lines(args.train_folder, args.resume, args.overrides, reeval_variants):
            print("\t".join([service] + overrides))
    elif args.cmd == "upload":
        upload(args.train_folder, args.overrides)


if __name__ == "__main__":
    main()
