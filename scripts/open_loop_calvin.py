#!/usr/bin/env python3
"""Open-loop per-timestep loss of one CALVIN training run on the held-out validation split
(env D demonstrations, never trained on), once per modality combo.

Every sample is one window of ExtendedDiskDataset's language index -- an observation
frame inside an annotated subtask window plus its 10-step action chunk -- annotated with
its subtask (task, instruction), position in the window (offset, progress) and a gripper
phase (flower/evaluation/open_loop.py). Two losses per sample, with noise keyed on the
frame so every model and combo sees identical draws:

  fm_t<t>  flow-matching loss (the training objective) at fixed flow time t, mean over
           the chunk; fm_mean is their mean over the t grid
  mse_k<k> squared error of the Euler-sampled chunk (the rollout path) at chunk step k;
           mse_mean is their mean

A dropout run is scored on every token combo (eval_pipeline.modality_combos), a plain run
on all inputs only. Each combo is written atomically to
<train_folder>/open_loop/<ckpt>/parts/<combo>.csv.gz (an interrupted run resumes at the
next missing combo, and --combos can split one model across processes); once every combo
is present they are merged into <train_folder>/open_loop/<ckpt>/calvin_val.csv.gz.

Usage:
  python scripts/open_loop_calvin.py --train-folder <dir> [--combos 0,3] [--max-windows N]
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))
from eval_pipeline import modality_combos  # noqa: E402
from mimicgen_pipeline import _resolve  # noqa: E402
from open_loop_analysis import combo_label  # noqa: E402

from flower.evaluation.eval_records import checkpoint_name  # noqa: E402
from flower.evaluation.open_loop import (  # noqa: E402
    annotation_batches,
    fm_loss_per_sample,
    sample_noise,
    sampled_mse_per_sample,
    set_combo,
    window_index,
)
from flower.evaluation.utils import get_default_mode_and_env  # noqa: E402

DATASET_PATH = "/calvin_data/task_ABCD_D"
EVAL_CFG_OVERWRITE = {
    "datamodule": {"datasets": {"lang_dataset": {"lang_folder": "lang_annotations"}}},
    "model": {"num_sampling_steps": 4},
}


def model_label(train_cfg) -> str:
    model = train_cfg.model
    if not model.get("modality_dropout", False):
        return "no dropout"
    return f"{model.get('modality_dropout_strategy', 'dirichlet')} keep{model.modality_dropout_keep_fraction:g}"


def combo_flags(combo: Dict[str, bool]) -> Dict[str, bool]:
    return {"use_static": combo["rgb_static"], "use_wrist": combo["rgb_gripper"], "use_lang": combo["language"]}


def to_device(batch, device):
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    return batch


def sample_index(dataset) -> pd.DataFrame:
    """window_index for the validation dataset, checked against its own lookup."""
    lang = np.load(dataset.abs_datasets_dir / dataset.lang_folder / "auto_lang_ann.npy", allow_pickle=True).item()
    rows = dataset.extracted_ep_npz_name_to_npy_idx
    gripper = dataset.extracted_ep_rel_actions[:, -1]
    index = window_index(
        lang["info"]["indx"],
        lang["language"]["task"],
        lang["language"]["ann"],
        dataset.min_window_size,
        lambda frames: gripper[[rows[int(f)] for f in frames]],
    )
    assert len(index) == len(dataset) and (index["frame"].values == dataset.episode_lookup).all()
    return index


@torch.inference_mode()
def score_combo(model, loader, index, combo, t_grid, seed, device) -> pd.DataFrame:
    set_combo(model, combo)
    K = len(t_grid)
    out = []
    for batch in loader:
        idx = batch["idx"].numpy()
        batch = to_device(batch, device)
        actions = batch["actions"]
        if actions.dim() == 4:
            actions = actions.squeeze(1)
        noise = sample_noise(index["frame"].values[idx], K + 1, tuple(actions.shape[1:]), seed).to(device)
        cond = model.encode_observations(batch)
        fm = fm_loss_per_sample(model, cond, actions, t_grid, noise[:, :K]).mean(-1).cpu().numpy()
        mse = sampled_mse_per_sample(model, cond, actions, noise[:, K]).cpu().numpy()
        part = {"idx": idx}
        part.update({f"fm_t{t:g}": fm[:, k] for k, t in enumerate(t_grid)})
        part["fm_mean"] = fm.mean(1)
        part.update({f"mse_k{k}": mse[:, k] for k in range(mse.shape[1])})
        part["mse_mean"] = mse.mean(1)
        out.append(pd.DataFrame(part))
    scores = pd.concat(out, ignore_index=True)
    return index.merge(scores, on="idx", how="inner")


def write_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-folder", required=True)
    parser.add_argument("--checkpoint", default=None, help="default: <run>/seed_<seed>/saved_models/last.ckpt")
    parser.add_argument("--combos", default=None, help="comma-separated combo indices (default: all)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-windows", type=int, default=None, help="only the first N annotated windows (smoke)")
    parser.add_argument("--t-grid", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir", default=None, help="default: <train_folder>/open_loop/<ckpt>[_smoke<N>]"
    )
    args = parser.parse_args()

    train_cfg = OmegaConf.load(Path(args.train_folder) / ".hydra" / "config.yaml")
    overrides = [f"checkpoint={args.checkpoint}"] if args.checkpoint else []
    train_folder, checkpoint = _resolve(args.train_folder, overrides)
    # A --max-windows smoke never lands where the full run's parts are resumed from.
    suffix = f"_smoke{args.max_windows}" if args.max_windows is not None else ""
    out_dir = Path(args.out_dir or Path(train_folder) / "open_loop" / f"{checkpoint_name(checkpoint)}{suffix}")
    t_grid = [float(t) for t in args.t_grid.split(",")]

    if train_cfg.model.get("modality_dropout", False):
        combos = modality_combos(bool(train_cfg.model.use_proprio))
    else:
        combos = modality_combos(False)[:1]
    selected = [int(i) for i in args.combos.split(",")] if args.combos else range(len(combos))
    todo = [i for i in selected if not (out_dir / "parts" / f"{combo_label(combo_flags(combos[i]))}.csv.gz").exists()]
    print(f"{train_folder}: {len(combos)} combos, {len(todo)} to score -> {out_dir}")

    if todo:
        model, _, data_module, _ = get_default_mode_and_env(
            train_folder,
            DATASET_PATH,
            checkpoint,
            prep_dm_and_deps=False,
            device_id=0,
            eval_cfg_overwrite=EVAL_CFG_OVERWRITE,
        )
        model.num_sampling_steps = EVAL_CFG_OVERWRITE["model"]["num_sampling_steps"]
        model.eval()
        device = next(model.parameters()).device
        data_module.setup()
        dataset = data_module.val_datasets["lang"]
        index = sample_index(dataset)
        if args.max_windows is not None:
            index = index[index["ann_idx"] < args.max_windows]
        loader = DataLoader(
            Subset(dataset, index["idx"].tolist()),
            batch_sampler=annotation_batches(index["ann_idx"].tolist(), args.batch_size),
            num_workers=args.num_workers,
            pin_memory=True,
        )
        meta = {
            "run": str(Path(train_folder).relative_to(Path(train_folder).parents[1])),
            "model": model_label(train_cfg),
            "modality_dropout": bool(train_cfg.model.get("modality_dropout", False)),
            "keep_fraction": train_cfg.model.get("modality_dropout_keep_fraction", None)
            if train_cfg.model.get("modality_dropout", False)
            else None,
            "strategy": train_cfg.model.get("modality_dropout_strategy", "dirichlet"),
            "checkpoint": checkpoint_name(checkpoint),
        }
        for i in todo:
            combo = combos[i]
            flags = combo_flags(combo)
            label = combo_label(flags)
            start = time.time()
            df = score_combo(model, loader, index, combo, t_grid, args.seed, device)
            df = pd.concat([pd.DataFrame({**meta, "combo": label, **flags}, index=df.index), df], axis=1)
            write_atomic(df, out_dir / "parts" / f"{label}.csv.gz")
            print(f"  {label}: {len(df)} samples in {time.time() - start:.0f}s, mse_mean {df['mse_mean'].mean():.4f}")

    parts = [out_dir / "parts" / f"{combo_label(combo_flags(c))}.csv.gz" for c in combos]
    if all(p.exists() for p in parts):
        write_atomic(pd.concat([pd.read_csv(p) for p in parts], ignore_index=True), out_dir / "calvin_val.csv.gz")
        print(f"Merged {len(parts)} combos into {out_dir / 'calvin_val.csv.gz'}")


if __name__ == "__main__":
    main()
