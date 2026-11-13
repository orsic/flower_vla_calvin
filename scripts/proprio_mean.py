#!/usr/bin/env python3
"""Mean proprio conditioning vector of a trained FLOWERVLA over its training data, for the
eval_modalities.proprio_state=false masked-input eval (FLOWERVLA.eval_proprio_fill).

Withholding proprio zeroes stateless_norm(encode_proprio(robot_obs)), the term dit_forward
adds to the conditioning; masking it puts this mean in its place instead. Each sample's
vector has zero mean and unit variance over its dit_dim entries (squared norm dit_dim), so
energy_share = ||mean||^2 / dit_dim is the part of it the average already carries.

Writes <run>/eval_logs/<ckpt>/proprio_mean.pt (eval_records.proprio_mean_path) as
{"mean", "samples", "norm", "energy_share"}; does nothing for a run without proprio or
when the file exists. Driven by ./run.sh proprio-mean and PIPELINE_MASKED=1.

Usage:
  python scripts/proprio_mean.py --train-folder <dir> [--checkpoint <ckpt>] [--samples 4096]
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))
from mimicgen_pipeline import _resolve  # noqa: E402

from flower.evaluation.eval_records import proprio_mean_path  # noqa: E402


def target(train_folder: str, checkpoint: Optional[str]) -> Optional[Path]:
    """The file to write, or None when there is nothing to do."""
    train_cfg = OmegaConf.load(Path(train_folder) / ".hydra" / "config.yaml")
    if not bool(train_cfg.model.get("use_proprio", False)):
        print(f"{train_folder}: model.use_proprio=False, no proprio to average")
        return None
    _, checkpoint = _resolve(train_folder, [f"checkpoint={checkpoint}"] if checkpoint else [])
    path = proprio_mean_path(checkpoint)
    if path.exists():
        print(f"{path} exists")
        return None
    return path


def compute(train_folder: str, checkpoint: str, samples: int) -> dict:
    import hydra
    import torch

    from flower.evaluation.utils import load_mode_from_safetensor
    from flower.models.networks.transformers import stateless_norm

    torch.manual_seed(0)
    model = load_mode_from_safetensor(Path(checkpoint)).to("cuda").eval()
    cfg = OmegaConf.load(Path(train_folder) / ".hydra" / "config.yaml")
    cfg.num_workers = 0  # container shared memory is too small for loader workers
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup("fit")
    # Index the training dataset directly: DataLoader workers die on the container's small /dev/shm.
    dataset = next(iter(dm.train_datasets.values()))
    indices = torch.randperm(len(dataset))[:samples].tolist()
    embs = []
    with torch.no_grad():
        for start in range(0, len(indices), 256):
            obs = torch.stack([torch.as_tensor(dataset[i]["robot_obs"]).float() for i in indices[start : start + 256]])
            B = obs.shape[0]
            e = model.encode_proprio(obs.to(model.device), torch.ones(B, 10, 7), (B,))
            embs.append(stateless_norm(e.float()).cpu())
    n = torch.cat(embs)
    mean = n.mean(0)
    return {
        "mean": mean,
        "samples": len(n),
        "norm": float(mean.norm()),
        "energy_share": float(mean.norm() ** 2 / n.shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-folder", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--samples", type=int, default=4096)
    args = parser.parse_args()

    path = target(args.train_folder, args.checkpoint)
    if path is None:
        return
    import torch

    _, checkpoint = _resolve(args.train_folder, [f"checkpoint={args.checkpoint}"] if args.checkpoint else [])
    result = compute(args.train_folder, checkpoint, args.samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(result, tmp)
    os.replace(tmp, path)
    print(f"{path}: {result['samples']} samples, ||mean|| {result['norm']:.3f}, energy share {result['energy_share']:.4f}")


if __name__ == "__main__":
    main()
