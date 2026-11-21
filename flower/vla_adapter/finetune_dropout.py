"""Fine-tune VLA-Adapter with FlowerVLA's whole-modality dropout (model.modality_dropout_strategy=subset).

Wraps upstream VLA-Adapter/vla-scripts/finetune.py unchanged (its config, data pipeline,
LoRA, DDP, optimizer and checkpointing): only its run_forward_pass is wrapped. During
training, each sample keeps a non-empty subset of (static camera, wrist camera, language,
proprio), drawn as in FlowerVLA (flower.models.networks.modality_dropout.
sample_modality_subsets: weight 2^|S|), and the dropped inputs are masked out of the LLM
and the action head at unchanged positions (flower/vla_adapter/masked_forward.py, the same
masks the evaluation uses for removed inputs). Validation and --modality_dropout none run
upstream's forward as is.

--resume_from <run>/<run id>--<step>_chkpt continues a run from one of its checkpoints:
upstream's --resume only reloads the action head and proprio projector and rebuilds the VLA
from the base VLM, so the VLA weights (merged LoRA + action queries, the checkpoint's
model.safetensors) are served in place of the base VLM, both as the starting point and as
the base each save merges the new LoRA into. Pass the original run's flags unchanged:
--num_steps_before_decay is shifted so the LR decay lands at the same total step
(upstream's scheduler counts from the resume; --max_steps and checkpoint names count total
steps). Upstream saves no optimizer state, so AdamW restarts from zero moments.

Usage (cwd-independent; torchrun as for upstream):
  torchrun --standalone --nnodes 1 --nproc-per-node N flower/vla_adapter/finetune_dropout.py \
      --modality_dropout subset [--resume_from DIR] <upstream finetune.py flags...>
"""

import importlib.util
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Tuple, Union

import torch

VLA_ADAPTER = Path(__file__).resolve().parents[2] / "VLA-Adapter"
STRATEGIES = ("none", "subset")
MODALITIES = ("static", "wrist", "lang", "proprio")
REPORT_EVERY = 1000  # forward passes between kept-subset frequency reports (rank 0)
# finetune.py's replace_map: base-VLM key fragments it renames to the HF VLA's names.
UPSTREAM_RENAMES = (
    ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
    ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
    ("llm_backbone.llm", "language_model"),
    ("projector.projector.0", "projector.fc1"),
    ("projector.projector.2", "projector.fc2"),
    ("projector.projector.4", "projector.fc3"),
    ("gamma", "scale_factor"),
)


def pop_flag(argv: List[str], name: str, default: str) -> Tuple[str, List[str]]:
    """Remove `--name value` / `--name=value` from argv; returns (value, remaining argv)."""
    rest, value, i = [], default, 0
    while i < len(argv):
        arg = argv[i]
        if arg == f"--{name}":
            value, i = argv[i + 1], i + 2
            continue
        if arg.startswith(f"--{name}="):
            value = arg.split("=", 1)[1]
        else:
            rest.append(arg)
        i += 1
    return value, rest


def checkpoint_step(checkpoint: Union[str, Path]) -> int:
    """The training step of a <run id>--<step>_chkpt dir holding its head and projector."""
    checkpoint = Path(checkpoint)
    match = re.search(r"--(\d+)_chkpt$", checkpoint.name)
    if not match:
        raise ValueError(f"{checkpoint}: expected a <run id>--<step>_chkpt checkpoint dir")
    step = int(match.group(1))
    for name in ("action_head", "proprio_projector"):
        if not (checkpoint / f"{name}--{step}_checkpoint.pt").is_file():
            raise FileNotFoundError(checkpoint / f"{name}--{step}_checkpoint.pt")
    return step


def resume_argv(argv: List[str], checkpoint: Union[str, Path]) -> List[str]:
    """Upstream flags resuming from `checkpoint`: --num_steps_before_decay shifted by the
    step (at least 0, i.e. already decayed), plus upstream's own --resume flags."""
    step = checkpoint_step(checkpoint)
    out, shifted, i = [], False, 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--num_steps_before_decay":
            out += [arg, str(max(int(argv[i + 1]) - step, 0))]
            shifted, i = True, i + 2
            continue
        if arg.startswith("--num_steps_before_decay="):
            arg, shifted = f"--num_steps_before_decay={max(int(arg.split('=', 1)[1]) - step, 0)}", True
        out.append(arg)
        i += 1
    if not shifted:
        raise ValueError("--resume_from needs the run's --num_steps_before_decay to place the LR decay")
    return out + ["--resume", "True", "--resume_step", str(step), "--resum_vla_path", str(checkpoint)]


def checkpoint_vlm(checkpoint: Union[str, Path]) -> Callable:
    """Stand-in for prismatic.models.load in finetune.py: a "VLM" whose state_dict() is the
    checkpoint's merged VLA weights, which already carry the names finetune.py renames to."""
    path = Path(checkpoint) / "model.safetensors"

    def load(vlm_path, hf_token=None, load_for_training=True):
        from safetensors.torch import load_file

        state = load_file(str(path))
        return SimpleNamespace(state_dict=lambda: state)

    return load


def without_flash_attention() -> None:
    """Load the base VLM's Qwen2.5 without FlashAttention-2 (flash_attn isn't in the image).
    upstream only reads its weights there (finetune.py copies them into the VLA it builds
    from pretrained_models/configs, which uses the default attention), so they are the same."""
    from prismatic.models.backbones.llm import qwen25

    init = qwen25.Qwen25LLMBackbone.__init__

    def init_without_flash(self, *args, use_flash_attention_2: bool = False, **kwargs):
        init(self, *args, use_flash_attention_2=use_flash_attention_2, **kwargs)

    qwen25.Qwen25LLMBackbone.__init__ = init_without_flash


def load_upstream():
    """Import vla-scripts/finetune.py as a module. Upstream resolves pretrained_models/
    configs relative to the VLA-Adapter root, so that becomes the working directory."""
    sys.path.insert(0, str(VLA_ADAPTER))
    os.chdir(VLA_ADAPTER)
    without_flash_attention()
    spec = importlib.util.spec_from_file_location("vla_adapter_finetune", VLA_ADAPTER / "vla-scripts" / "finetune.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dropout_forward_pass(original: Callable, strategy: str, report: Callable[[str], None] = print) -> Callable:
    """run_forward_pass with whole-modality dropout on training batches."""
    from flower.models.networks.modality_dropout import sample_modality_subsets
    from flower.vla_adapter import masked_forward

    if strategy not in STRATEGIES:
        raise ValueError(f"--modality_dropout must be one of {STRATEGIES}, got {strategy!r}")
    kept = {"n": 0, "samples": 0, "counts": torch.zeros(len(MODALITIES)), "full": 0}

    def run_forward_pass(vla, action_head, proprio_projector, batch, *args, **kwargs):
        if strategy == "none" or not vla.training:
            return original(vla, action_head, proprio_projector, batch, *args, **kwargs)
        labels, attention_mask = batch["labels"], batch["attention_mask"]
        keep = sample_modality_subsets(labels.shape[0], len(MODALITIES), device=labels.device)
        masks = masked_forward.drop_masks(keep, labels, attention_mask)
        model, head = masked_forward.unwrap(vla), getattr(action_head, "module", action_head)
        with masked_forward.injected_masks(model, head, masks):
            out = original(vla, action_head, proprio_projector, batch, *args, **kwargs)
        kept["n"] += 1
        kept["samples"] += keep.shape[0]
        kept["counts"] += keep.float().sum(0).cpu()
        kept["full"] += int(keep.all(1).sum())
        if kept["n"] % REPORT_EVERY == 0 and int(os.environ.get("RANK", "0")) == 0:
            freq = kept["counts"] / kept["samples"]
            report(
                "modality dropout (subset), kept fraction: "
                + ", ".join(f"{m} {f:.3f}" for m, f in zip(MODALITIES, freq.tolist()))
                + f", all inputs {kept['full'] / kept['samples']:.3f}"
            )
        return out

    return run_forward_pass


def main(argv: List[str]) -> None:
    strategy, rest = pop_flag(argv, "modality_dropout", "subset")
    resume_from, rest = pop_flag(rest, "resume_from", "")
    if resume_from:
        rest = resume_argv(rest, resume_from)
    upstream = load_upstream()
    upstream.run_forward_pass = dropout_forward_pass(upstream.run_forward_pass, strategy)
    if resume_from:
        upstream.load = checkpoint_vlm(resume_from)
    sys.argv = [sys.argv[0]] + rest
    upstream.finetune()


if __name__ == "__main__":
    main(sys.argv[1:])
