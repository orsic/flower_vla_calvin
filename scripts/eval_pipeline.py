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
  - Either way, plus one more "eval-plus" line per single withheld modality
    (MODALITY_OFF_VARIANTS: rgb_gripper/rgb_static/proprio/language off in turn, the
    other three on) -- each writing to its own plus_<bench>_no_<modality>/result.csv via
    an explicit csv_dir= override, since perturbation_sr.py/severity_sr.py both assume
    exactly one modality combo per LIBERO-Plus result.csv. The proprio variant is never
    emitted for a model.use_proprio=False checkpoint: flower_eval_libero.py's
    EvaluateLibero raises on eval_modalities.proprio=False there (there is no
    proprioception to withhold), so planning it would crash the eval container, not just
    waste a duplicate run. --skip-modality-off (PIPELINE_SKIP_MODALITY_OFF=1) suppresses
    all 4 of these lines, e.g. to keep a chained ./run.sh train-dropout -> pipeline run
    from taking ~5x longer on its LIBERO-Plus portion; catch them up later with
    `--reeval --reeval-suites plus_<bench>_no_<modality>`.

--overrides (after a literal "--") are Hydra overrides appended, unmodified, to every
emitted line -- last on the line, so they take precedence (see run.sh's run_train/eval
helpers for the same last-wins convention). They are also inspected for the handful of
keys (benchmark_name, checkpoint, train_folder, csv_dir) that change which result.csv
`plan --resume` and `upload` read, so overriding e.g. benchmark_name redirects both. Do not
pass eval_modalities.* or csv_dir through -- the first fights the combo sweep, the second
collapses every LIBERO-Plus suite (including all 4 modality-off ones) into one file.

`upload` runs scripts/pid_modality.py on the LIBERO (orig) result.csv and
scripts/severity_sr.py on the LIBERO-Plus result.csv and on each present modality-off
result.csv, then attaches all of that (up to 2 + 2*4 = 10 files, plus the 2 raw orig/plus
result.csv's -- 12 total) to a W&B artifact on the *training* run (same project/entity/id
as training_libero.py's setup_logger derives from the run dir); every member is optional
and simply omitted when its source result.csv doesn't exist yet. Since `plan --resume`
treats every combo already in a result.csv as done, `PIPELINE_RESUME=1 ./run.sh pipeline
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

# Variant token -> (eval_modalities key to switch off, result-directory/artifact-member
# suffix). Suffixes reuse compare_eval_csvs.MODALITY_COLUMNS' short names
# (wrist/static/proprio/lang) rather than a second vocabulary. Iteration order is the
# canonical plan/report order (3rd-person, then 1st-person, then proprio, then
# language), here and in analyze_wandb.py.
MODALITY_OFF_VARIANTS = {
    "plus_no_wrist": ("rgb_gripper", "no_wrist"),
    "plus_no_static": ("rgb_static", "no_static"),
    "plus_no_proprio": ("proprio", "no_proprio"),
    "plus_no_lang": ("language", "no_lang"),
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


def modality_off_combo(variant: str) -> Dict[str, bool]:
    """full_modality_combo() with exactly the one modality MODALITY_OFF_VARIANTS[variant]
    names switched off."""
    key, _suffix = MODALITY_OFF_VARIANTS[variant]
    combo = full_modality_combo()
    combo[key] = False
    return combo


def all_variants(use_proprio: bool) -> List[str]:
    """Every result-directory variant a run can be evaluated under, in plan order.
    "plus_no_proprio" is absent when use_proprio is False: EvaluateLibero.__init__
    raises on eval_modalities.proprio=False there (flower_eval_libero.py) -- it is
    structurally not applicable to such a run, not an eval that merely hasn't run yet."""
    variants = ["orig", "plus"] + list(MODALITY_OFF_VARIANTS)
    if not use_proprio:
        variants.remove("plus_no_proprio")
    return variants


def suite_dir_name(variant: str, benchmark_name: str) -> str:
    """Result-directory name for one variant. orig/plus keep eval_records.result_dir's
    "<variant>_<suite>" layout; a modality-off suite is "plus_<suite>_no_<modality>" --
    the benchmark has to stay in the middle since benchmark_name= (the LIBERO
    benchmark-registry key) can't itself carry the modality suffix, so that suffix can
    only live in the directory name."""
    if variant in MODALITY_OFF_VARIANTS:
        _key, suffix = MODALITY_OFF_VARIANTS[variant]
        return f"plus_{benchmark_name}_{suffix}"
    return f"{variant}_{benchmark_name}"


def modality_off_member(variant: str) -> str:
    """W&B artifact member name for one modality-off result.csv."""
    _key, suffix = MODALITY_OFF_VARIANTS[variant]
    return f"libero_plus_{suffix}.csv"


def modality_off_severity_member(variant: str) -> str:
    """W&B artifact member name for one modality-off severity_sr.csv -- distinct from
    the full-modality suite's "severity_sr.csv" even though both files are literally
    named severity_sr.csv on disk (they live in different suite directories)."""
    _key, suffix = MODALITY_OFF_VARIANTS[variant]
    return f"severity_sr_{suffix}.csv"


def _use_proprio(train_folder: str) -> bool:
    """model.use_proprio from <train_folder>/.hydra/config.yaml."""
    return bool(OmegaConf.load(Path(train_folder) / ".hydra" / "config.yaml").model.use_proprio)


def parse_overrides(overrides: List[str]) -> Dict[str, str]:
    """"key=value" tokens -> dict, last occurrence wins (mirrors Hydra's own semantics)."""
    parsed = {}
    for override in overrides:
        key, sep, value = override.partition("=")
        if sep:
            parsed[key] = value
    return parsed


def _csv_path_for_suite(train_folder: str, checkpoint: str, suite_dir: str) -> Path:
    """Same layout as eval_records.result_dir, without its mkdir side effect (plan/upload
    only read this path, never create it)."""
    return Path(train_folder) / "eval_logs" / checkpoint_name(checkpoint) / suite_dir / "result.csv"


def _csv_path(train_folder: str, checkpoint: str, variant: str, benchmark_name: str) -> Path:
    return _csv_path_for_suite(train_folder, checkpoint, suite_dir_name(variant, benchmark_name))


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


def resolve_reeval_variants(
    suites_arg: Optional[str], benchmark_name: str, use_proprio: bool = True
) -> Set[str]:
    """Parse --reeval-suites into a set of variants ("orig"/"plus"/one of
    MODALITY_OFF_VARIANTS) to re-evaluate.

    suites_arg is None when --reeval-suites wasn't passed at all -- re-evaluate every
    suite this run can have. When passed, each comma-separated token must name a result
    directory exactly (e.g. "plus_libero_10", "plus_libero_10_no_static"), checked
    against the *resolved* benchmark_name so a trailing benchmark_name= override is
    honored. A token for a different benchmark is almost certainly a mistake, not
    something to silently ignore -- but a token that names a real suite this
    (use_proprio=False) run simply can't have (plus_<bench>_no_proprio) is a milder case:
    skipped with a note, not an error, unless it's the only suite named.
    """
    valid = {suite_dir_name(v, benchmark_name): v for v in all_variants(use_proprio)}
    if suites_arg is None:
        return set(valid.values())
    tokens = [token.strip() for token in suites_arg.split(",") if token.strip()]
    if not tokens:
        raise ValueError(
            f"--reeval-suites was given but named no suite -- expected a comma-separated "
            f"list from {sorted(valid)}"
        )
    not_applicable = {suite_dir_name(v, benchmark_name) for v in all_variants(True)} - set(valid)
    variants: Set[str] = set()
    for token in tokens:
        if token in not_applicable:
            print(
                f"reeval: skipping {token!r} -- this checkpoint has model.use_proprio=False, "
                "so there is no proprioception to withhold",
                file=sys.stderr,
            )
            continue
        if token not in valid:
            raise ValueError(
                f"--reeval-suites: {token!r} is not a result directory for benchmark "
                f"{benchmark_name!r} -- expected one of {sorted(valid)}"
            )
        variants.add(valid[token])
    if not variants:
        raise ValueError(
            f"--reeval-suites named no suite that applies to this run -- expected one of {sorted(valid)}"
        )
    return variants


def rotate_reeval_csvs(
    variants: Set[str],
    parsed: Dict[str, str],
    resolved_train_folder: str,
    checkpoint: str,
    benchmark_name: str,
    use_proprio: bool = True,
) -> List[Optional[Path]]:
    """Back up each selected variant's result.csv so plan_lines starts it from empty.

    Must be called once, before plan_lines -- see the module docstring for why rotating
    per-invocation or after planning both silently lose data.
    """
    csv_dir_override = parsed.get("csv_dir")
    if csv_dir_override and variants != set(all_variants(use_proprio)):
        raise ValueError(
            f"csv_dir={csv_dir_override} makes every suite share one result.csv; "
            "--reeval-suites cannot rotate one without discarding the others. Select "
            "every suite, or drop csv_dir."
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
    skip_modality_off: bool = False,
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

    if not skip_modality_off:
        for variant in all_variants(use_proprio):
            if variant not in MODALITY_OFF_VARIANTS:
                continue
            if reeval_variants is not None and variant not in reeval_variants:
                continue
            combo = modality_off_combo(variant)
            if already_done(variant, combo):
                continue
            csv_dir = _csv_path(resolved_train_folder, checkpoint, variant, benchmark_name).parent
            overrides = _eval_overrides(benchmark_name, resolved_train_folder, checkpoint, combo)
            overrides.append(f"csv_dir={csv_dir}")
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


def modality_off_csv_paths(
    train_folder: str, checkpoint: str, benchmark_name: str, use_proprio: bool
) -> Dict[str, Path]:
    """variant -> result.csv path, for every modality-off suite this run can have --
    plus_no_proprio is absent from the dict entirely when use_proprio is False (see
    all_variants)."""
    return {
        variant: _csv_path(train_folder, checkpoint, variant, benchmark_name)
        for variant in all_variants(use_proprio)
        if variant in MODALITY_OFF_VARIANTS
    }


def artifact_members(
    orig_csv: Path,
    plus_csv: Path,
    modality_off_csvs: Dict[str, Path],
    pid_txt: Optional[Path],
) -> List[Tuple[Path, str]]:
    """(file, artifact member name) for every derived/raw file that actually exists.
    Every member is optional: a suite that hasn't been evaluated yet, or whose severity
    computation failed (missing LIBERO-Plus assets), is simply left out of the artifact
    rather than failing the upload -- analyze_wandb.py reports it as missing instead."""
    members: List[Tuple[Path, str]] = []
    if orig_csv.exists():
        members.append((orig_csv, "libero_orig.csv"))
    if plus_csv.exists():
        members.append((plus_csv, "libero_plus.csv"))
    if pid_txt is not None:
        members.append((pid_txt, "pid_modality.txt"))
    severity_csv = write_severity_csv(plus_csv, orig_csv)
    if severity_csv is not None:
        members.append((severity_csv, "severity_sr.csv"))
    for variant, csv_path in modality_off_csvs.items():
        if not csv_path.exists():
            continue
        members.append((csv_path, modality_off_member(variant)))
        # Same call as the full-modality suite above -- severity_sr keys the paired
        # baseline on (modality_combo, base_task), so a dropout run's orig result.csv
        # (which holds all 14 combos) supplies this same ablation's unperturbed LIBERO
        # numbers here for free. A non-dropout run's orig (1 combo only) just yields
        # orig_n=0 for this variant -- expected, not an error.
        modality_off_severity_csv = write_severity_csv(csv_path, orig_csv)
        if modality_off_severity_csv is not None:
            members.append((modality_off_severity_csv, modality_off_severity_member(variant)))
    return members


def upload(train_folder: str, extra_overrides: List[str]) -> None:
    train_cfg = OmegaConf.load(Path(train_folder) / ".hydra" / "config.yaml")
    benchmark_name, resolved_train_folder, checkpoint = _resolve(train_folder, extra_overrides)
    use_proprio = _use_proprio(train_folder)

    orig_csv = _csv_path(resolved_train_folder, checkpoint, "orig", benchmark_name)
    plus_csv = _csv_path(resolved_train_folder, checkpoint, "plus", benchmark_name)
    modality_off_csvs = modality_off_csv_paths(resolved_train_folder, checkpoint, benchmark_name, use_proprio)

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

    members = artifact_members(orig_csv, plus_csv, modality_off_csvs, pid_txt)

    if not any(p.exists() for p in [orig_csv, plus_csv, *modality_off_csvs.values()]):
        print(f"Nothing to upload: no result.csv found under {orig_csv.parent.parent}.")
        return

    run_id = wandb_run_id(resolved_train_folder)
    run = wandb.init(
        project=str(train_cfg.logger.project),
        entity=str(train_cfg.logger.entity),
        id=run_id,
        resume="allow",
    )
    artifact = wandb.Artifact(artifact_name(run_id), type="evaluation")
    for path, name in members:
        artifact.add_file(str(path), name=name)
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
    plan_p.add_argument("--skip-modality-off", action="store_true")
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
            use_proprio = _use_proprio(args.train_folder)
            try:
                reeval_variants = resolve_reeval_variants(args.reeval_suites, benchmark_name, use_proprio)
                # Rotation must happen here, before plan_lines: see the module docstring
                # and plan_lines' own docstring for why the reverse order silently loses
                # data (resume would read the not-yet-rotated file and plan nothing).
                rotate_reeval_csvs(
                    reeval_variants, parsed, resolved_train_folder, checkpoint, benchmark_name, use_proprio
                )
            except ValueError as exc:
                sys.exit(str(exc))

        for service, overrides in plan_lines(
            args.train_folder, args.resume, args.overrides, reeval_variants, args.skip_modality_off
        ):
            print("\t".join([service] + overrides))
    elif args.cmd == "upload":
        upload(args.train_folder, args.overrides)


if __name__ == "__main__":
    main()
