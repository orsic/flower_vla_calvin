#!/usr/bin/env python3
"""Machine-readable tables of every open-loop CALVIN result (scripts/open_loop_calvin.py).

Reads every merged calvin_val.csv.gz (default: all under /saves/train_logs/calvin_abcd_d*/)
and writes, one row per run and cell (combo, masked) unless noted:

  open_loop_by_run.csv           n, mean / sd / 95% CI of mse_mean and fm_mean, and their
                                 paired increase over the run's reference cell (mse_delta,
                                 fm_delta)
  open_loop_by_run_timestep.csv  the same per 5-frame bin of the subtask (tbin); bins with
                                 fewer than 200 samples are dropped, as in the figures
  open_loop_by_run_phase.csv     per gripper phase
  open_loop_by_run_task.csv      per task
  open_loop_by_model.csv         per model and cell: mean and sd over the model's runs of
                                 each per-run mean, with the runs and seeds

A run's reference cell is its unmasked combo with the most inputs: all inputs for a full
model, the trained subset for a model trained without one input. Deltas are empty for a
run scored on a single cell. Results written before use_proprio / masked existed read as
neither (no such run had proprio or masking).

Usage:
  python scripts/open_loop_tables.py [--csv a.csv.gz ...] [--out-dir DIR] [--bin-width 5]
"""

import argparse
import glob
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml

from open_loop_analysis import DEFAULT_GLOB, MIN_BIN_COUNT, _mean_ci, timestep_bin

META = ["run", "model", "seed"]
CELL = ["combo", "masked", "use_static", "use_wrist", "use_lang", "use_proprio"]
METRICS = ["mse_mean", "fm_mean", "mse_delta", "fm_delta"]


def seed_of(run_dir: Path) -> Optional[int]:
    cfg = run_dir / ".hydra" / "config.yaml"
    if not cfg.exists():
        return None
    return (yaml.safe_load(cfg.read_text()) or {}).get("seed")


def load_runs(paths: List[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["seed"] = seed_of(Path(path).parents[2])  # <run>/open_loop/<ckpt>/calvin_val.csv.gz
        frames.append(df)
    if not frames:
        raise SystemExit(f"No open-loop CSVs found ({paths or DEFAULT_GLOB}).")
    df = pd.concat(frames, ignore_index=True)
    df["masked"] = df["masked"].fillna("") if "masked" in df else ""
    df["use_proprio"] = df["use_proprio"].fillna(False).astype(bool) if "use_proprio" in df else False
    df["seed"] = df["seed"].astype("Int64")
    return df


def reference_combo(run_df: pd.DataFrame) -> str:
    """The run's unmasked combo with the most inputs."""
    cells = run_df.loc[run_df["masked"] == "", ["combo", "use_static", "use_wrist", "use_lang", "use_proprio"]]
    cells = cells.drop_duplicates("combo")
    n_inputs = cells[["use_static", "use_wrist", "use_lang", "use_proprio"]].astype(int).sum(axis=1)
    return str(cells.loc[n_inputs.idxmax(), "combo"])


def with_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """mse_delta / fm_delta: per sample, the cell's loss minus the run's reference cell's."""
    parts = []
    for _, run_df in df.groupby("run", sort=False):
        run_df = run_df.copy()
        if run_df["combo"].nunique() > 1:
            ref = run_df[run_df["combo"] == reference_combo(run_df)].set_index("idx")
            for metric in ("mse_mean", "fm_mean"):
                run_df[metric.replace("_mean", "_delta")] = run_df[metric].values - ref[metric].reindex(run_df["idx"]).values
        else:
            run_df["mse_delta"] = np.nan
            run_df["fm_delta"] = np.nan
        parts.append(run_df)
    return pd.concat(parts, ignore_index=True)


def summarize(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    """Per run, cell and `by`: n plus mean, sd and 95% CI of every metric."""
    groups = df.groupby(META + CELL + by, dropna=False, sort=False)
    out = groups.size().rename("n").to_frame()
    for metric in METRICS:
        agg = _mean_ci(groups[metric])[["mean", "std", "ci"]]
        out = out.join(agg.add_prefix(f"{metric}_"))
    return out.reset_index()


def by_model(runs: pd.DataFrame) -> pd.DataFrame:
    """Per model and cell: mean and sd over runs of each per-run mean."""
    rows = []
    for key, g in runs.groupby(["model"] + CELL, dropna=False, sort=False):
        row = dict(zip(["model"] + CELL, key))
        seeds = sorted(int(s) for s in g["seed"].dropna().unique())
        row.update(runs=len(g), seeds=" ".join(str(s) for s in seeds))
        for metric in METRICS:
            values = g[f"{metric}_mean"].dropna()
            row[metric] = values.mean() if len(values) else np.nan
            row[f"{metric}_sd"] = values.std(ddof=1) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_tables(df: pd.DataFrame, out_dir: Path, bin_width: int = 5) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = with_deltas(df)
    df["tbin"] = timestep_bin(df["offset"], bin_width)
    runs = summarize(df, [])
    timestep = summarize(df, ["tbin"])
    tables = {
        "open_loop_by_run": runs,
        "open_loop_by_run_timestep": timestep[timestep["n"] >= MIN_BIN_COUNT],
        "open_loop_by_run_phase": summarize(df, ["phase"]),
        "open_loop_by_run_task": summarize(df, ["task"]),
        "open_loop_by_model": by_model(runs),
    }
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False, float_format="%.6g")
    print(f"Wrote open-loop tables for {df['run'].nunique()} runs to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", nargs="*", default=None, help=f"default: {DEFAULT_GLOB}")
    parser.add_argument("--out-dir", default="/saves/plots/open_loop_tables")
    parser.add_argument("--bin-width", type=int, default=5, help="timestep bin width in frames")
    args = parser.parse_args()
    paths = args.csv if args.csv else sorted(p for p in glob.glob(DEFAULT_GLOB) if "_smoke" not in p)
    write_tables(load_runs(paths), Path(args.out_dir), args.bin_width)


if __name__ == "__main__":
    main()
