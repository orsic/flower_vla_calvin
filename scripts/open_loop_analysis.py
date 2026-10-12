#!/usr/bin/env python3
"""Per-timestep / per-model / per-modality-combo breakdown of the open-loop CALVIN
validation losses written by scripts/open_loop_calvin.py.

Reads one or more calvin_val.csv.gz (default: every one under
/saves/train_logs/calvin_abcd_d*/*/open_loop/) and writes PDFs + summary CSVs. The
x-axis of the line plots is the timestep along the demonstration: frames since the start
of the annotated subtask (CALVIN's language-annotated windows, 34-65 frames), binned.

  loss_vs_timestep_<metric>.pdf   one panel per model, one line per combo
  delta_vs_timestep_<metric>.pdf  same, as the paired per-sample increase over the
                                  model's own all-inputs loss ("how much does dropping
                                  modalities hurt at this point of the demonstration");
                                  all inputs is the zero line
  combo_vs_timestep_<metric>.pdf  one panel per combo, one line per model
  phase_<metric>.pdf              combo x gripper-phase heatmap per model
  task_<metric>.pdf               task x combo heatmap per model
  summary_by_{timestep,phase,task}.csv

Usage:
  python scripts/open_loop_analysis.py [--csv a.csv.gz b.csv.gz ...] [--out-dir DIR]
"""

import argparse
import glob
from pathlib import Path
from typing import Dict, List, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ALL = "static+wrist+lang"
COMBO_ORDER = [ALL, "static+wrist", "static+lang", "wrist+lang", "static", "wrist", "lang"]
COMBO_COLORS = {
    ALL: "#4a4a4a",
    "static+wrist": "#2a78d6",
    "static+lang": "#e8710a",
    "wrist+lang": "#1a9e5a",
    "static": "#9b59b6",
    "wrist": "#c0392b",
    "lang": "#b8a000",
}
PHASE_ORDER = ["approach", "grasp", "hold", "release", "retreat", "no_grasp"]
# mse_mean: squared error of the action chunk the policy samples (as in a rollout) against
# the demonstrated actions. fm_mean: the training objective (flow-matching loss).
METRICS = {"mse_mean": "predicted-action MSE", "fm_mean": "training (flow-matching) loss"}
XLABEL = "timestep in subtask (frames since its start)"
# Bins with fewer samples per (model, combo) are not drawn: only the 65-frame windows
# reach the last timesteps.
MIN_BIN_COUNT = 200
DEFAULT_GLOB = "/saves/train_logs/calvin_abcd_d*/*/open_loop/*/calvin_val.csv.gz"


def combo_label(flags: Mapping[str, bool]) -> str:
    """"static+wrist+lang"-style name of a combo from its use_* flags."""
    parts = [name for name in ("static", "wrist", "lang") if bool(flags[f"use_{name}"])]
    return "+".join(parts)


def timestep_bin(offset: pd.Series, width: int = 5) -> pd.Series:
    """Start frame of the `width`-frame bin holding each timestep (0, 5, 10, ...)."""
    return (offset // width) * width


def paired_delta(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per (run, idx): metric(combo) - metric(all inputs), all inputs included (as 0).
    Only runs scored on more than all inputs; keeps the other columns of the combo's row."""
    multi = df.groupby("run")["combo"].transform("nunique") > 1
    base = df.loc[multi & (df["combo"] == ALL), ["run", "idx", metric]].rename(columns={metric: "_base"})
    out = df.loc[multi].merge(base, on=["run", "idx"], how="inner")
    out["delta"] = out[metric] - out["_base"]
    return out.drop(columns="_base")


def _mean_ci(group: pd.core.groupby.SeriesGroupBy) -> pd.DataFrame:
    """Mean and normal-approximation 95% half-width. Samples overlap within a subtask
    window, so this understates the true uncertainty -- read it as a relative error bar."""
    agg = group.agg(["mean", "std", "count"])
    agg["ci"] = 1.96 * agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    return agg


def _models(df: pd.DataFrame) -> List[str]:
    return list(dict.fromkeys(df.sort_values(["model_order", "model"])["model"]))


def _combos(df: pd.DataFrame) -> List[str]:
    return [c for c in COMBO_ORDER if c in set(df["combo"])]


def _save(fig, path: Path, **kwargs) -> None:
    """PDF for print plus a PNG twin (same stem) for markdown, e.g. docs/results.md."""
    fig.savefig(path, **kwargs)
    fig.savefig(Path(path).with_suffix(".png"), dpi=150, **kwargs)
    plt.close(fig)


def _line_panels(df, value, panel_col, panels, line_col, lines, colors, ylabel, title, path, legend_title, width):
    # At most 4 panels per row, so 7 combos still get readable panels.
    ncols = min(len(panels), 4)
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows + 0.6), sharey=True, squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax, panel in zip(axes.flat, panels):
        sub = df[df[panel_col] == panel]
        for line in lines:
            s = sub[sub[line_col] == line]
            if s.empty:
                continue
            agg = _mean_ci(s.groupby("tbin")[value])
            agg = agg[agg["count"] >= MIN_BIN_COUNT]
            x = agg.index + (width - 1) / 2
            style = {"ls": "--", "lw": 1.2} if line == ALL else {}
            ax.plot(x, agg["mean"], marker="o", ms=3, color=colors.get(line), label=line, **style)
            ax.fill_between(x, agg["mean"] - agg["ci"], agg["mean"] + agg["ci"], color=colors.get(line), alpha=0.15)
        ax.set_title(panel, fontsize=10)
        ax.set_xlabel(XLABEL)
        ax.grid(alpha=0.3)
    for row in axes:
        row[0].set_ylabel(ylabel)
    # One legend for the whole figure, under the panels (every panel shares the colors).
    handles, labels = {}, {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    order = [lab for lab in lines if lab in handles]
    fig.legend(
        [handles[lab] for lab in order], order, title=legend_title, loc="lower center",
        ncol=max(len(order), 1), frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.6 / (3.4 * nrows + 0.6) + 0.06, 1, 1))
    _save(fig, path)


def _heatmaps(table_by_model: Dict[str, pd.DataFrame], title: str, path: Path, cbar_label: str) -> None:
    models = list(table_by_model)
    first = next(iter(table_by_model.values()))
    h = max(3.0, 0.28 * len(first.index) + 1.5)
    w = max(3.5, 0.75 * len(first.columns) + 1.5)
    # Room between panels for each panel's own row labels (task names are long).
    label_w = 0.07 * max(len(str(i)) for i in first.index)
    fig, axes = plt.subplots(
        1, len(models), figsize=((w + label_w) * len(models), h), squeeze=False,
        gridspec_kw={"wspace": label_w / w + 0.15},
    )
    vmax = max(float(np.nanmax(np.abs(t.values))) for t in table_by_model.values()) or 1.0
    for ax, model in zip(axes[0], models):
        t = table_by_model[model]
        im = ax.imshow(t.values, cmap="Reds", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(t.columns)), t.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(t.index)), t.index, fontsize=7)
        if len(t.index) * len(t.columns) <= 60:
            for (i, j), v in np.ndenumerate(t.values):
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6)
        ax.set_title(model, fontsize=10)
    fig.colorbar(im, ax=axes[0].tolist(), label=cbar_label, shrink=0.8)
    fig.suptitle(title)
    _save(fig, path, bbox_inches="tight")


def load(paths: List[str]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    if not frames:
        raise SystemExit(f"No open-loop CSVs found ({paths or DEFAULT_GLOB}).")
    return pd.concat(frames, ignore_index=True)


def analyze(df: pd.DataFrame, out_dir: Path, bin_width: int = 5) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["tbin"] = timestep_bin(df["offset"], bin_width)
    # Least dropout first, the no-dropout baseline last.
    df["model_order"] = np.where(df["modality_dropout"], -df["keep_fraction"].fillna(1.0), 1.0)
    models, combos = _models(df), _combos(df)
    model_colors = {m: plt.cm.viridis(i / max(len(models) - 1, 1)) for i, m in enumerate(models)}

    summaries = {"timestep": [], "phase": [], "task": []}
    for metric, metric_name in METRICS.items():
        _line_panels(
            df, metric, "model", models, "combo", combos, COMBO_COLORS, metric_name,
            f"{metric_name} along the held-out demonstrations (CALVIN D validation), per model",
            out_dir / f"loss_vs_timestep_{metric}.pdf", "inputs given to the model", bin_width,
        )
        _line_panels(
            df, metric, "combo", combos, "model", models, model_colors, metric_name,
            f"{metric_name} along the held-out demonstrations, per input combination",
            out_dir / f"combo_vs_timestep_{metric}.pdf", "model", bin_width,
        )

        delta = paired_delta(df, metric)
        phases = [p for p in PHASE_ORDER if p in set(df["phase"])]
        tasks = sorted(df["task"].unique())
        _heatmaps(
            {
                m: df[df.model == m].pivot_table(index="combo", columns="phase", values=metric, aggfunc="mean")
                .reindex(index=combos, columns=phases)
                for m in models
            },
            f"mean {metric_name} by gripper phase",
            out_dir / f"phase_{metric}.pdf",
            metric_name,
        )
        _heatmaps(
            {
                m: df[df.model == m].pivot_table(index="task", columns="combo", values=metric, aggfunc="mean")
                .reindex(index=tasks, columns=combos)
                for m in models
            },
            f"mean {metric_name} by task",
            out_dir / f"task_{metric}.pdf",
            metric_name,
        )
        if not delta.empty:
            _line_panels(
                delta, "delta", "model", _models(delta), "combo", _combos(delta), COMBO_COLORS,
                f"Δ {metric_name} vs all inputs",
                f"increase in {metric_name} from withholding inputs, along the demonstrations",
                out_dir / f"delta_vs_timestep_{metric}.pdf", "inputs given to the model", bin_width,
            )

        for key, by in (("timestep", "tbin"), ("phase", "phase"), ("task", "task")):
            agg = _mean_ci(df.groupby(["model", "combo", by])[metric]).add_prefix(f"{metric}_")
            if not delta.empty:
                dagg = _mean_ci(delta.groupby(["model", "combo", by])["delta"]).add_prefix(f"{metric}_delta_")
                agg = agg.join(dagg, how="left")
            summaries[key].append(agg)

    for key, parts in summaries.items():
        pd.concat(parts, axis=1).reset_index().to_csv(out_dir / f"summary_by_{key}.csv", index=False)

    print(f"Wrote figures and summaries to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", nargs="*", default=None, help=f"default: {DEFAULT_GLOB}")
    parser.add_argument("--out-dir", default="/saves/plots/open_loop")
    parser.add_argument("--bin-width", type=int, default=5, help="timestep bin width in frames")
    args = parser.parse_args()
    paths = args.csv if args.csv else sorted(p for p in glob.glob(DEFAULT_GLOB) if "_smoke" not in p)
    analyze(load(paths), Path(args.out_dir), args.bin_width)


if __name__ == "__main__":
    main()
