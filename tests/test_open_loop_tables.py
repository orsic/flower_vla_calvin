"""Tests for scripts/open_loop_tables.py: machine-readable tables of every open-loop result.

- Each run's reference cell is its unmasked combo with the most inputs (all inputs, or a
  specialist's trained subset); deltas are paired per sample against it.
- Results written before use_proprio / masked existed read as neither.
- The per-model table aggregates over runs (mean and sd of the per-run means).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from open_loop_tables import by_model, load_runs, reference_combo, summarize, with_deltas  # noqa: E402


def _rows(run, model, combo, flags, mse, masked=None, n=4):
    out = pd.DataFrame(
        {
            "run": run, "model": model, "combo": combo, "idx": np.arange(n), "offset": np.arange(n),
            "phase": "hold", "task": "open_drawer", "mse_mean": np.full(n, mse, dtype=float) + np.arange(n),
            "fm_mean": np.full(n, 2 * mse, dtype=float),
            **dict(zip(("use_static", "use_wrist", "use_lang"), flags[:3])),
        }
    )
    if len(flags) == 4:
        out["use_proprio"] = flags[3]
    if masked is not None:
        out["masked"] = masked
    return out


def _write(tmp_path, group, ts, seed, frames):
    run_dir = tmp_path / group / ts
    (run_dir / ".hydra").mkdir(parents=True)
    if seed is not None:
        (run_dir / ".hydra" / "config.yaml").write_text(f"seed: {seed}\n")
    path = run_dir / "open_loop" / "last" / "calvin_val.csv.gz"
    path.parent.mkdir(parents=True)
    pd.concat(frames).to_csv(path, index=False, compression="gzip")
    return str(path)


@pytest.fixture
def df(tmp_path):
    paths = [
        # an older dropout run: no use_proprio / masked columns
        _write(tmp_path, "calvin_abcd_d_dropout", "a", 42, [
            _rows("d/a", "dirichlet keep0.5", "static+wrist+lang", (True, True, True), 1.0),
            _rows("d/a", "dirichlet keep0.5", "static", (True, False, False), 3.0),
        ]),
        # a proprio model with a masked cell and proprio zeroed
        _write(tmp_path, "calvin_abcd_d", "p", 100, [
            _rows("p/p", "no dropout + proprio", "static+wrist+lang+proprio", (True, True, True, True), 1.0, ""),
            _rows("p/p", "no dropout + proprio", "wrist+lang+proprio (static masked)", (False, True, True, True), 2.0, "static"),
            _rows("p/p", "no dropout + proprio", "static+wrist+lang", (True, True, True, False), 1.5, ""),
        ]),
        # a specialist, its trained subset only; no .hydra seed
        _write(tmp_path, "calvin_abcd_d_static+lang+proprio", "s", None, [
            _rows("s/s", "no dropout + proprio (trained without wrist)", "static+lang+proprio", (True, False, True, True), 1.2, ""),
        ]),
    ]
    return load_runs(paths)


def test_load_fills_older_columns_and_reads_seeds(df):
    old = df[df.run == "d/a"]
    assert (old["masked"] == "").all() and not old["use_proprio"].any()
    seeds = df.groupby("run")["seed"].first()
    assert seeds["d/a"] == 42 and seeds["p/p"] == 100 and pd.isna(seeds["s/s"])


def test_reference_cell_per_kind_of_run(df):
    assert reference_combo(df[df.run == "d/a"]) == "static+wrist+lang"
    assert reference_combo(df[df.run == "p/p"]) == "static+wrist+lang+proprio"
    assert reference_combo(df[df.run == "s/s"]) == "static+lang+proprio"


def test_deltas_are_paired_against_the_reference_cell(df):
    d = with_deltas(df)
    masked = d[d.combo == "wrist+lang+proprio (static masked)"]
    assert masked["mse_delta"].tolist() == [1.0] * 4
    assert (d[d.combo == "static+wrist+lang+proprio"]["mse_delta"] == 0).all()
    assert d[d.run == "s/s"]["mse_delta"].isna().all()


def test_summarize_and_model_aggregate(df):
    runs = summarize(with_deltas(df), [])
    row = runs[(runs.run == "p/p") & (runs.combo == "static+wrist+lang")].iloc[0]
    assert row["n"] == 4 and row["mse_mean_mean"] == pytest.approx(3.0) and row["mse_delta_mean"] == pytest.approx(0.5)
    two = pd.concat([runs, runs.assign(run="p/q", seed=5, mse_mean_mean=runs["mse_mean_mean"] + 1.0)])
    models = by_model(two[two.model == "no dropout + proprio"])
    full = models[models.combo == "static+wrist+lang+proprio"].iloc[0]
    assert full["runs"] == 2 and full["seeds"] == "5 100"
    assert full["mse_mean"] == pytest.approx(3.0) and full["mse_mean_sd"] == pytest.approx(np.std([2.5, 3.5], ddof=1))
