"""Print the LIBERO tables of docs/results.md from the W&B evaluation artifacts that
scripts/results_libero_collect.py downloaded.

Only the main-study runs in MODELS are used: trained with ffba3d2 or later (W&B project
multimodal_florence) and evaluated after ca969a8 (checked on every CSV's eval_timestamp,
except the multi-GPU runs in NO_TS, whose shards leave it empty). Values are success
rates in %, mean ± sd over a model's seeds.

    python scripts/results_libero_tables.py [--artifacts /saves/libero_artifacts]
"""

import argparse
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

CUT = "2026-09-23_14-46-43"  # ca969a8, 2026-09-23 16:46:43 +0200, in the CSVs' UTC timestamps
PROJECT = "multimodal_florence"
MODELS: Dict[str, Dict[int, str]] = {
    "no dropout": {
        42: "libero_10_2026-09-18_16-47-38",
        5: "libero_10_2026-09-18_16-47-48",
        100: "libero_10_2026-09-18_16-48-01",
    },
    "no dropout + proprio": {
        42: "libero_10_static+wrist+lang+proprio_2026-09-18_16-47-17",
        5: "libero_10_static+wrist+lang+proprio_2026-09-18_16-47-22",
        100: "libero_10_static+wrist+lang+proprio_2026-09-18_16-47-49",
    },
    "dropout, no proprio": {
        42: "libero_10_dropout_2026-09-14_17-14-47",
        5: "libero_10_dropout_2026-09-14_17-15-19",
        100: "libero_10_dropout_2026-11-02_12-18-49",
    },
    "dropout, proprio keep 1.0": {
        42: "libero_10_dropout_2026-09-14_17-00-43",
        5: "libero_10_dropout_2026-09-14_17-07-42",
        100: "libero_10_dropout_2026-09-14_17-08-28",
    },
    "dropout, proprio keep 0.75": {
        42: "libero_10_dropout_2026-09-14_17-12-09",
        5: "libero_10_dropout_2026-09-14_17-03-21",
        100: "libero_10_dropout_2026-09-14_17-12-40",
    },
    "dropout, proprio keep 0.5": {
        42: "libero_10_dropout_2026-09-14_17-00-58",
        5: "libero_10_dropout_2026-09-14_17-07-23",
        100: "libero_10_dropout_2026-09-14_17-08-24",
    },
    "dropout, proprio keep 0.25": {
        42: "libero_10_dropout_2026-09-15_16-05-08",
        5: "libero_10_dropout_2026-09-15_16-11-38",
        100: "libero_10_dropout_2026-11-02_12-18-36",
    },
}
# Whole-modality dropout (model.modality_dropout_strategy=subset) with proprioception,
# evaluated with all inputs or one withheld (PIPELINE_SKIP_MULTI_OFF=1): its own section,
# since the main tables cover every input combination.
SUBSET_RUNS: Dict[int, str] = {
    42: "libero_10_dropout_2026-11-04_21-49-24",
    5: "libero_10_dropout_2026-11-04_21-51-57",
    100: "libero_10_dropout_2026-11-07_12-40-49",
}
# No dropout with proprioception, trained without one input (model.modalities.<x>=False),
# keyed by that input; evaluated only within their trained inputs.
FIXED_RUNS: Dict[str, Dict[int, str]] = {
    "static": {
        42: "libero_10_wrist+lang+proprio_2026-11-06_15-31-06",
        5: "libero_10_wrist+lang+proprio_2026-11-06_15-31-13",
    },
    "wrist": {
        42: "libero_10_static+lang+proprio_2026-11-06_15-30-31",
        5: "libero_10_static+lang+proprio_2026-11-06_15-30-36",
    },
    "lang": {
        42: "libero_10_static+wrist+proprio_2026-11-06_15-30-50",
        5: "libero_10_static+wrist+proprio_2026-11-06_15-30-57",
    },
}
# Multi-GPU evaluations leave eval_timestamp empty (rank shards are written without it);
# their validity was checked by file modification time or training date instead.
NO_TS = {
    "libero_10_2026-09-18_16-48-01",
    "libero_10_dropout_2026-09-15_16-11-38",
    "libero_10_static+wrist+lang+proprio_2026-09-18_16-47-49",
    "libero_10_dropout_2026-11-02_12-18-49",
    "libero_10_dropout_2026-11-02_12-18-36",
}
PROPRIO = {m: ("proprio" in m and "no proprio" not in m) for m in MODELS}
NAMES = (
    ("static", "use_rgb_static"),
    ("wrist", "use_rgb_gripper"),
    ("lang", "use_language"),
    ("proprio", "use_proprio"),
)
TOKEN_COMBOS = ["static+wrist+lang", "static+wrist", "static+lang", "wrist+lang", "static", "wrist", "lang"]
CATS = [
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
]
CAT_TASKS = {
    "Camera Viewpoints": 419,
    "Robot Initial States": 393,
    "Light Conditions": 274,
    "Sensor Noise": 449,
    "Objects Layout": 312,
}
COUNTS = ["successes", "n", "orig_successes", "orig_n"]
# Physical-size tables: (category, title, severity_sr bin -> row label or None to skip,
# whether the rows form a range that the doubled-range tables merge).
GROUPS: Dict[str, tuple] = {
    "cam_rot": (
        "Camera Viewpoints",
        "Camera Viewpoints, camera rotation",
        lambda b: b if b.startswith("rot") else None,
        True,
    ),
    "cam_scale": (
        "Camera Viewpoints",
        "Camera Viewpoints, camera distance scale",
        lambda b: b if b.startswith("scale") else None,
        True,
    ),
    "robot": ("Robot Initial States", "Robot Initial States, initial joint offset", lambda b: b, True),
    "light": ("Light Conditions", "Light Conditions, light severity", lambda b: b, True),
    "noise_level": (
        "Sensor Noise",
        "Sensor Noise by level, pooled over the 5 noise types",
        lambda b: "level " + b.rsplit("_", 1)[1],
        True,
    ),
    "noise_type": (
        "Sensor Noise",
        "Sensor Noise by type, pooled over the 10 levels",
        lambda b: b.rsplit("_", 1)[0],
        False,
    ),
    "obj_disp": (
        "Objects Layout",
        "Objects Layout, target displacement",
        lambda b: b if b.startswith("displacement") else None,
        True,
    ),
    "obj_dist": (
        "Objects Layout",
        "Objects Layout, number of distractors",
        lambda b: b if b.startswith("distractors") else None,
        True,
    ),
}


def merge_bins(bins: Sequence[str]) -> List[List[str]]:
    """Pair adjacent bins from the low end, doubling each range; an odd last bin joins
    the previous group (so 5 bins give [1, 2], [3, 4, 5])."""
    groups = [list(bins[i : i + 2]) for i in range(0, len(bins), 2)]
    if len(groups) > 1 and len(groups[-1]) == 1:
        last = groups.pop()
        groups[-1] += last
    return groups


def span_label(labels: Sequence[str]) -> str:
    """'rot~0-15deg' + 'rot~15-30deg' -> 'rot~0-30deg'; '0.1rad' + '0.2rad' -> '0.1-0.2rad'."""
    if len(labels) == 1:
        return labels[0]
    num = r"\d+(?:\.\d+)?"
    first = list(re.finditer(num, labels[0]))
    hi = re.findall(num, labels[-1])[-1]
    if len(first) == 1:
        m = first[0]
        return labels[0][: m.start()] + f"{m.group()}-{hi}" + labels[0][m.end() :]
    m = first[-1]
    return labels[0][: m.start()] + hi + labels[0][m.end() :]


def merged_counts(d: pd.DataFrame, groups: List[List[str]], labels: List[str]) -> pd.DataFrame:
    """Sum the COUNTS columns of d (indexed by bin) over each group of bins."""
    return pd.DataFrame([d.loc[g, COUNTS].sum() for g in groups], index=labels)


def ms(v: Sequence[float]) -> str:
    v = 100 * np.array(v, dtype=float)
    return f"{v.mean():.1f}" if len(v) == 1 else f"{v.mean():.1f} ± {v.std(ddof=1):.1f}"


def md(df: pd.DataFrame) -> str:
    cols = [df.index.name or ""] + list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join([str(i)] + [str(x) for x in r]) + " |" for i, r in df.iterrows()]
    return "\n".join(out)


def full(m: str) -> str:
    return "static+wrist+lang+proprio" if PROPRIO[m] else "static+wrist+lang"


class Artifacts:
    def __init__(self, root: Path) -> None:
        self.root = root / PROJECT

    def path(self, run: str, member: str) -> Path:
        return self.root / run / member

    def load(self, run: str, member: str) -> pd.DataFrame:
        d = pd.read_csv(self.path(run, member))
        if d["eval_timestamp"].isna().all():
            assert run in NO_TS, (run, member)
        else:
            assert d["eval_timestamp"].min() >= CUT, (run, member)
        d["combo"] = d.apply(lambda r: "+".join(n for n, c in NAMES if int(r[c])), axis=1)
        return d


def size_table(
    sev: Dict[str, Dict[int, pd.DataFrame]],
    cat: str,
    lab: Callable[[str], Optional[str]],
    doubled: bool,
) -> pd.DataFrame:
    """One perturbation-size table: per bin (or doubled group), task count, mean ± sd
    over seeds per model (one column per key of sev), and the paired original-LIBERO
    success pooled over all runs."""
    rows: Dict[str, List[pd.DataFrame]] = {}
    order: List[str] = []
    for m in sev:
        per_seed = []
        for d in sev[m].values():
            d = d[d.category == cat].copy()
            d["row"] = d.bin.map(lab)
            d = d.dropna(subset=["row"])
            g = d.groupby("row", sort=False)[COUNTS].sum()
            per_seed.append(g)
            order = list(dict.fromkeys(order + list(g.index)))
        rows[m] = per_seed
    if doubled:
        groups = merge_bins(order)
        order = [span_label(g) for g in groups]
        rows = {m: [merged_counts(g, groups, order) for g in ps] for m, ps in rows.items()}
    ref = next(iter(rows.values()))[0]
    allruns = sum(g[["orig_successes", "orig_n"]] for ps in rows.values() for g in ps)
    t = {}
    for r in order:
        t[r] = {"tasks": int(ref.loc[r, "n"])}
        for m, ps in rows.items():
            assert all(int(g.loc[r, "n"]) == t[r]["tasks"] for g in ps), (cat, r, m)
            t[r][m] = ms([g.loc[r, "successes"] / g.loc[r, "n"] for g in ps])
        t[r]["orig, paired"] = f"{100 * allruns.loc[r, 'orig_successes'] / allruns.loc[r, 'orig_n']:.0f}"
    return pd.DataFrame(t).T.rename_axis("bin")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--artifacts", type=Path, default=Path("/saves/libero_artifacts"))
    a = Artifacts(ap.parse_args().artifacts)

    orig = {m: {s: a.load(r, "libero_orig.csv") for s, r in runs.items()} for m, runs in MODELS.items()}
    plus = {m: {s: a.load(r, "libero_plus.csv") for s, r in runs.items()} for m, runs in MODELS.items()}
    for m in MODELS:
        for s, d in orig[m].items():
            assert d.groupby("combo").size().eq(200).all(), (m, s)
        for s, d in plus[m].items():
            assert len(d) == 2519 and d.combo.nunique() == 1 and d.combo.iloc[0] == full(m), (m, s)

    print("## Headline: LIBERO-10 and LIBERO-Plus with all inputs (%)")
    rows = {}
    for m, runs in MODELS.items():
        o = [orig[m][s].pipe(lambda d: d[d["combo"] == full(m)]).success.mean() for s in runs]
        p = [plus[m][s].success.mean() for s in runs]
        rows[m] = {
            "seeds": " / ".join(map(str, runs)),
            "LIBERO-10 per seed": " / ".join(f"{100 * x:.1f}" for x in o),
            "LIBERO-10": ms(o),
            "LIBERO-Plus per seed": " / ".join(f"{100 * x:.1f}" for x in p),
            "LIBERO-Plus": ms(p),
        }
    print(md(pd.DataFrame(rows).T.rename_axis("model")))

    print("\n## LIBERO-10 by modality combo (%), mean over seeds")
    order = []
    for c in TOKEN_COMBOS:
        order += [c + "+proprio", c]
    t = {}
    for m in MODELS:
        vals = {}
        for c in order:
            v = [d.pipe(lambda d: d[d["combo"] == c]).success.mean() for d in orig[m].values() if (d.combo == c).any()]
            vals[c] = ms(v) if v else "n/a"
        t[m] = vals
    tab = pd.DataFrame(t).rename_axis("inputs")
    print(md(tab[~(tab == "n/a").all(axis=1)]))

    print("\n## LIBERO-Plus by category, all inputs (%), mean over seeds")
    t = {}
    for m in MODELS:
        t[m] = {
            c: ms([d.pipe(lambda d: d[d["task_category"] == c]).success.mean() for d in plus[m].values()]) for c in CATS
        }
        t[m]["all"] = ms([d.success.mean() for d in plus[m].values()])
    print(md(pd.DataFrame(t).rename_axis("category")))
    print("tasks per category:", plus["no dropout"][42].task_category.value_counts().to_dict())

    print("\n## LIBERO-Plus by difficulty level, all inputs (%), mean over seeds")
    t = {
        m: {
            lv: ms([d.pipe(lambda d: d[d["difficulty_level"] == lv]).success.mean() for d in plus[m].values()])
            for lv in range(1, 6)
        }
        for m in MODELS
    }
    print(md(pd.DataFrame(t).T.rename_axis("model")))
    print("tasks per level:", plus["no dropout"][42].difficulty_level.value_counts().sort_index().to_dict())

    print(
        "\n## LIBERO-Plus with one modality withheld at evaluation (%), mean over seeds; LIBERO-10 same combo in brackets"
    )
    t = {}
    for m, runs in MODELS.items():
        vals = {"all inputs": ms([d.success.mean() for d in plus[m].values()])}
        for v in ("lang", "static", "wrist", "proprio"):
            if v == "proprio" and not PROPRIO[m]:
                vals["no " + v] = "n/a"
                continue
            ds = [a.load(r, f"libero_plus_no_{v}.csv") for r in runs.values()]
            assert all(len(d) == 2519 for d in ds)
            c = ds[0].combo.iloc[0]
            o = [
                orig[m][s].pipe(lambda d: d[d["combo"] == c]).success.mean()
                for s in runs
                if (orig[m][s].combo == c).any()
            ]
            vals["no " + v] = ms([d.success.mean() for d in ds]) + (f" ({ms(o)})" if o else "")
        t[m] = vals
    print(md(pd.DataFrame(t).T.rename_axis("model")))

    print("\n## LIBERO-Plus no_* per category (%), mean over seeds")
    for v in ("lang", "static", "wrist", "proprio"):
        t = {}
        for m, runs in MODELS.items():
            if v == "proprio" and not PROPRIO[m]:
                continue
            ds = [a.load(r, f"libero_plus_no_{v}.csv") for r in runs.values()]
            t[m] = {c: ms([d.pipe(lambda d: d[d["task_category"] == c]).success.mean() for d in ds]) for c in CATS}
        print(f"\n### no {v}\n" + md(pd.DataFrame(t).rename_axis("category")))

    print("\n## LIBERO-PRO, every run with pro_sr.csv (%)")
    for m, runs in MODELS.items():
        ps = {sd: pd.read_csv(a.path(r, "pro_sr.csv")) for sd, r in runs.items() if a.path(r, "pro_sr.csv").exists()}
        if not ps:
            continue
        nat = {sd: p[p.libero_variant == "pro"].set_index("perturbation").success_rate for sd, p in ps.items()}
        print(
            m,
            "seeds",
            list(ps),
            {
                k: ms([n[k] for n in nat.values()]) + " [" + " / ".join(f"{100 * n[k]:.1f}" for n in nat.values()) + "]"
                for k in ("lan", "object", "swap", "task")
            },
        )
        for sd, p in ps.items():
            for r in p[p.libero_variant == "pro_matched"].itertuples():
                print(
                    f"   seed {sd} matched {r.perturbation} {r.modality_combo} pro {100 * r.success_rate:.1f} "
                    f"orig {100 * r.orig_success_rate:.1f} delta {100 * r.delta:+.1f} p={r.mcnemar_p:.3g}"
                )

    sev = {
        m: {sd: pd.read_csv(a.path(r, "severity_sr.csv")).query("axis == 'severity'") for sd, r in runs.items()}
        for m, runs in MODELS.items()
    }
    print("\n## LIBERO-Plus by physical perturbation size, all inputs (%), mean over seeds")
    for cat, title, lab, _ in GROUPS.values():
        tab = size_table(sev, cat, lab, doubled=False)
        print(f"\n### {title} (tasks summed: {tab.tasks.sum()} of {CAT_TASKS[cat]})\n" + md(tab))

    print("\n## LIBERO-Plus by physical perturbation size, doubled ranges, all inputs (%), mean over seeds")
    for cat, title, lab, ranged in GROUPS.values():
        if not ranged:
            continue
        tab = size_table(sev, cat, lab, doubled=True)
        print(f"\n### {title} (tasks summed: {tab.tasks.sum()} of {CAT_TASKS[cat]})\n" + md(tab))

    print_subset(a)
    print_fixed(a)


def print_subset(a: Artifacts) -> None:
    """LIBERO-10, LIBERO-Plus and LIBERO-PRO for the subset-strategy runs (SUBSET_RUNS)."""
    seeds = list(SUBSET_RUNS)
    full = "static+wrist+lang+proprio"
    orig = {sd: a.load(r, "libero_orig.csv") for sd, r in SUBSET_RUNS.items()}
    plus = {sd: a.load(r, "libero_plus.csv") for sd, r in SUBSET_RUNS.items()}
    for sd in seeds:
        assert orig[sd].groupby("combo").size().eq(200).all() and orig[sd].combo.nunique() == 5, sd
        assert len(plus[sd]) == 2519 and set(plus[sd].combo) == {full}, sd

    def cell(vals: List[float]) -> str:
        return ms(vals) + " (" + " / ".join(f"{100 * v:.1f}" for v in vals) + ")"

    print(f"\n## Whole-modality dropout (subset), seeds {' / '.join(map(str, seeds))}: mean ± sd (per seed), %")
    print("\n### LIBERO-10 by inputs")
    combos = [c for c in orig[seeds[0]].combo.unique()]
    combos.sort(key=lambda c: (-c.count("+"), c))
    t = {c: {"LIBERO-10": cell([orig[sd][orig[sd].combo == c].success.mean() for sd in seeds])} for c in combos}
    print(md(pd.DataFrame(t).T.rename_axis("inputs")))

    print("\n### LIBERO-Plus, all inputs and one withheld")
    t = {"all inputs": cell([plus[sd].success.mean() for sd in seeds])}
    for v in ("lang", "static", "wrist", "proprio"):
        ds = [a.load(SUBSET_RUNS[sd], f"libero_plus_no_{v}.csv") for sd in seeds]
        assert all(len(d) == 2519 for d in ds)
        t["no " + v] = cell([d.success.mean() for d in ds])
    print(md(pd.DataFrame({"LIBERO-Plus": t}).rename_axis("inputs")))

    print("\n### LIBERO-Plus by category, all inputs")
    t = {c: {"subset": cell([plus[sd][plus[sd].task_category == c].success.mean() for sd in seeds])} for c in CATS}
    print(md(pd.DataFrame(t).T.rename_axis("category")))

    print("\n### LIBERO-Plus by category, one input withheld (mean over seeds)")
    t = {}
    for v in ("lang", "static", "wrist", "proprio"):
        ds = [a.load(SUBSET_RUNS[sd], f"libero_plus_no_{v}.csv") for sd in seeds]
        t["no " + v] = {c: ms([d[d.task_category == c].success.mean() for d in ds]) for c in CATS}
    print(md(pd.DataFrame(t).rename_axis("category")))

    print("\n### LIBERO-Plus by difficulty level, all inputs")
    t = {lv: {"subset": cell([plus[sd][plus[sd].difficulty_level == lv].success.mean() for sd in seeds])} for lv in range(1, 6)}
    print(md(pd.DataFrame(t).rename_axis("model")))

    # Perturbation size next to no dropout + proprio and the most robust Dirichlet model.
    compare = {
        "no dropout + proprio": MODELS["no dropout + proprio"],
        "dropout, proprio keep 1.0": MODELS["dropout, proprio keep 1.0"],
        "subset": SUBSET_RUNS,
    }
    sev = {
        m: {sd: pd.read_csv(a.path(r, "severity_sr.csv")).query("axis == 'severity'") for sd, r in runs.items()}
        for m, runs in compare.items()
    }
    for doubled in (False, True):
        print(f"\n### LIBERO-Plus by physical perturbation size{', doubled ranges' if doubled else ''}, all inputs")
        for cat, title, lab, ranged in GROUPS.values():
            if doubled and not ranged:
                continue
            tab = size_table(sev, cat, lab, doubled=doubled)
            print(f"\n#### {title} (tasks summed: {tab.tasks.sum()} of {CAT_TASKS[cat]})\n" + md(tab))

    print("\n### LIBERO-PRO")
    ps = {sd: pd.read_csv(a.path(r, "pro_sr.csv")) for sd, r in SUBSET_RUNS.items()}
    nat = {sd: p[p.libero_variant == "pro"].set_index("perturbation").success_rate for sd, p in ps.items()}
    print(md(pd.DataFrame({k: {"native, all inputs": cell([n[k] for n in nat.values()])} for k in ("lan", "object", "swap", "task")}).rename_axis("")))
    for sd, p in ps.items():
        for r in p[p.libero_variant == "pro_matched"].itertuples():
            print(
                f"   seed {sd} matched {r.perturbation} {r.modality_combo} pro {100 * r.success_rate:.1f} "
                f"orig {100 * r.orig_success_rate:.1f} delta {100 * r.delta:+.1f} p={r.mcnemar_p:.3g}"
            )



def print_fixed(a: Artifacts) -> None:
    """Runs trained without one input (FIXED_RUNS), next to the 3-seed Dirichlet keep 0.5 +
    proprio keep 0.5 model and the subset runs with the same input withheld at evaluation,
    and no dropout + proprio with all inputs."""
    dirichlet = MODELS["dropout, proprio keep 0.5"]
    ref = MODELS["no dropout + proprio"]
    for x, runs in FIXED_RUNS.items():
        trained = "+".join(n for n, _ in NAMES if n != x)
        seeds = list(runs)
        orig = {sd: a.load(r, "libero_orig.csv") for sd, r in runs.items()}
        plus = {sd: a.load(r, "libero_plus.csv") for sd, r in runs.items()}
        for sd in seeds:
            assert len(orig[sd]) == 200 and set(orig[sd].combo) == {trained}, sd
            assert len(plus[sd]) == 2519 and set(plus[sd].combo) == {trained}, sd

        def withheld(group: Dict[int, str]) -> List[pd.DataFrame]:
            return [a.load(r, f"libero_plus_no_{x}.csv") for r in group.values()]

        def on(group: Dict[int, str], combo: str) -> List[float]:
            return [a.load(r, "libero_orig.csv").pipe(lambda d: d[d.combo == combo]).success.mean() for r in group.values()]

        print(f"\n## No dropout + proprio trained without {x} ({trained}), seeds {' / '.join(map(str, seeds))}, %")
        cols = {
            f"trained without {x}": (
                [orig[sd].success.mean() for sd in seeds],
                [plus[sd].success.mean() for sd in seeds],
            ),
            f"Dirichlet pk0.5, {x} withheld": (on(dirichlet, trained), [d.success.mean() for d in withheld(dirichlet)]),
            f"subset, {x} withheld": (on(SUBSET_RUNS, trained), [d.success.mean() for d in withheld(SUBSET_RUNS)]),
            "no dropout + proprio, all inputs": (
                on(ref, "static+wrist+lang+proprio"),
                [a.load(r, "libero_plus.csv").success.mean() for r in ref.values()],
            ),
        }
        t = {c: {"LIBERO-10": ms(o), "LIBERO-Plus": ms(p), "per seed (Plus)": " / ".join(f"{100 * v:.1f}" for v in p)} for c, (o, p) in cols.items()}
        print(md(pd.DataFrame(t).T.rename_axis("model")))

        print("\n### LIBERO-Plus with a further input withheld")
        t = {"none": ms([plus[sd].success.mean() for sd in seeds])}
        for v in (n for n, _ in NAMES if n != x):
            ds = [a.load(r, f"libero_plus_no_{v}.csv") for r in runs.values()]
            assert all(len(d) == 2519 for d in ds)
            t["no " + v] = ms([d.success.mean() for d in ds])
        print(md(pd.DataFrame({"LIBERO-Plus": t}).rename_axis("also withheld")))

        print("\n### LIBERO-Plus by category")
        groups = {
            f"trained without {x}": list(plus.values()),
            f"Dirichlet pk0.5, {x} withheld": withheld(dirichlet),
            f"subset, {x} withheld": withheld(SUBSET_RUNS),
        }
        t = {c: {g: ms([d[d.task_category == c].success.mean() for d in ds]) for g, ds in groups.items()} for c in CATS}
        print(md(pd.DataFrame(t).T.rename_axis("category")))

        print("\n### LIBERO-PRO, native")
        nat = [pd.read_csv(a.path(r, "pro_sr.csv")).query("libero_variant == 'pro'").set_index("perturbation").success_rate for r in runs.values()]
        print(md(pd.DataFrame({k: {f"trained without {x}": ms([n[k] for n in nat])} for k in ("lan", "object", "swap", "task")}).rename_axis("")))


if __name__ == "__main__":
    main()
