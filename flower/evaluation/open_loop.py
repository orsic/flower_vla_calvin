"""Open-loop per-timestep loss on held-out demonstrations, per modality combo.

Pure helpers for scripts/open_loop_calvin.py. Every sample is one dataset window (one
observation frame + its action chunk) of an annotated CALVIN validation segment; its loss
is computed with the eval-time modality mask of one combo, at fixed flow times and with
noise keyed on the frame id, so every model and combo is scored on identical draws.
"""

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

TOKEN_KEYS = ("rgb_static", "rgb_gripper", "language")


def fm_loss_per_sample(
    model, cond: Dict, actions: torch.Tensor, t_grid: Sequence[float], noise: torch.Tensor
) -> torch.Tensor:
    """FLOWERVLA.rf_loss's squared velocity error at each t in `t_grid`, reduced over
    the action dimension only: [B, K, T]. `noise` is [B, K, T, D], one draw per t."""
    dtype = next(model.parameters()).dtype
    actions = actions.to(dtype)
    B = actions.size(0)
    losses = []
    for k, t_val in enumerate(t_grid):
        t = torch.full((B,), float(t_val), device=actions.device, dtype=dtype)
        z1 = noise[:, k].to(device=actions.device, dtype=dtype)
        zt = (1 - t.view(B, 1, 1)) * actions + t.view(B, 1, 1) * z1
        vtheta = model.dit_forward(zt, t, cond)
        losses.append((((z1 - actions) - vtheta) ** 2).float().mean(-1))
    return torch.stack(losses, dim=1)


def sampled_mse_per_sample(model, cond: Dict, actions: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Squared error of the Euler-sampled action chunk (the rollout path, from `noise`)
    against the demonstration, reduced over the action dimension: [B, T]."""
    dtype = next(model.parameters()).dtype
    pred = model.sample_actions(noise.to(device=actions.device, dtype=dtype), cond, inference=True)
    return ((pred.float() - actions.float()) ** 2).mean(-1)


def sample_noise(frame_ids: Sequence[int], n: int, shape: Tuple[int, ...], seed: int) -> torch.Tensor:
    """[B, n, *shape] standard normal noise, each row drawn from a generator seeded by
    (seed, frame id) -- independent of batch composition, model and combo."""
    rows = []
    for frame in frame_ids:
        gen = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(frame))
        rows.append(torch.randn((n, *shape), generator=gen))
    return torch.stack(rows)


def set_combo(model, combo: Dict[str, bool]) -> None:
    """Point the model's eval-time masks at one token combo. Unlike
    flower_evaluate.apply_eval_modalities, the all-on combo resets the mask to None, so
    combos can be swept on one loaded model."""
    keep = tuple(bool(combo[k]) for k in TOKEN_KEYS)
    if not any(keep):
        raise ValueError(f"combo must keep at least one token modality, got {combo}")
    model.eval_modality_mask = None if all(keep) else keep
    model.eval_proprio_mask = None


def gripper_phases(gripper_cmd: np.ndarray, margin: int = 2) -> List[str]:
    """Heuristic manipulation phase per frame from the gripper command (CALVIN: +1 open,
    -1 close). Frames within `margin` of an open->close switch are "grasp", of a
    close->open switch "release"; other closed frames "hold"; open frames before the
    first closed frame "approach", after it "retreat". A window that never closes is
    "no_grasp" throughout (pushes, sliders, buttons...)."""
    closed = np.asarray(gripper_cmd) < 0
    n = len(closed)
    if not closed.any():
        return ["no_grasp"] * n
    first_closed = int(np.argmax(closed))
    phases = ["hold" if c else ("approach" if i < first_closed else "retreat") for i, c in enumerate(closed)]
    for i in range(1, n):
        if closed[i] != closed[i - 1]:
            label = "grasp" if closed[i] else "release"
            for j in range(max(0, i - margin), min(n, i + margin + 1)):
                phases[j] = label
    return phases


def annotation_batches(ann_idx: Sequence[int], batch_size: int) -> List[List[int]]:
    """Positions 0..N-1 grouped into batches of at most `batch_size` consecutive samples of
    one annotation window. Every sample of a window has the same instruction, so no batch
    pads language: encode_observations pads text to the batch's longest instruction and
    the padding changes the conditioning, which would make a sample's loss depend on its
    batch neighbours (rollouts encode one instruction at a time, without padding)."""
    batches: List[List[int]] = []
    for pos, ann in enumerate(ann_idx):
        if batches and len(batches[-1]) < batch_size and ann_idx[batches[-1][0]] == ann:
            batches[-1].append(pos)
        else:
            batches.append([pos])
    return batches


def window_index(
    indx: Sequence[Tuple[int, int]],
    tasks: Sequence[str],
    anns: Sequence[str],
    min_window_size: int,
    gripper_cmd: Callable[[np.ndarray], np.ndarray],
) -> pd.DataFrame:
    """Per-sample annotation table, row i = dataset idx i of ExtendedDiskDataset's
    language index (same loop as DiskDataset._build_file_indices_lang, skip_frames=1,
    pretrain=False). `progress` is the sample's offset over the window's last valid
    start, in [0, 1]; `phase` is gripper_phases over the whole window [s, e], read at
    the sample's observation frame."""
    rows = []
    for ann_idx, (s, e) in enumerate(indx):
        frames = np.arange(s, e + 1)
        phases = gripper_phases(gripper_cmd(frames))
        starts = range(s, e + 1 - min_window_size)
        last = max(len(starts) - 1, 1)
        for offset, frame in enumerate(starts):
            rows.append(
                {
                    "ann_idx": ann_idx,
                    "task": tasks[ann_idx],
                    "ann_text": anns[ann_idx],
                    "win_start": s,
                    "win_end": e,
                    "frame": frame,
                    "offset": offset,
                    "progress": offset / last,
                    "phase": phases[frame - s],
                }
            )
    df = pd.DataFrame(rows)
    df.insert(0, "idx", np.arange(len(df)))
    return df
