"""forward_chunk against VLA-Adapter's own inference, on the real checkpoints (GPU only).

With all inputs, forward_chunk must reproduce experiments/robot/robot_utils.get_action
bit for bit, so masking support changes nothing in the all-inputs baseline. Every
single-input removal or masking must run and give a finite chunk that differs from it.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from flower.evaluation.vla_adapter_policy import Modalities, forward_chunk, load_vla_adapter

CKPT_ROOT = Path(os.environ.get("VLA_ADAPTER_CKPT_ROOT", "/saves/checkpoints/vla_adapter"))
CASES = {"libero": "LIBERO-Long-Pro", "calvin": "CALVIN-ABC-Pro"}
SPECS = [
    Modalities(static=False),
    Modalities(wrist=False),
    Modalities(lang=False),
    Modalities(proprio="off"),
    Modalities(mean_static=True),
    Modalities(mean_wrist=True),
    Modalities(blank_instruction=True),
    Modalities(proprio="mean"),
]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not all((CKPT_ROOT / name).is_dir() for name in CASES.values()),
    reason="needs a GPU and the VLA-Adapter checkpoints (./run.sh download-vla-adapter)",
)


def _obs(platform):
    rng = np.random.default_rng(0)
    static, wrist = ((256, 256), (256, 256)) if platform == "libero" else ((200, 200), (84, 84))
    return {
        "full_image": rng.integers(0, 255, (*static, 3), dtype=np.uint8),
        "wrist_image": rng.integers(0, 255, (*wrist, 3), dtype=np.uint8),
        "state": rng.normal(size=8) * 0.1,
    }


@pytest.fixture(scope="module", params=list(CASES))
def loaded(request):
    policy = load_vla_adapter(CKPT_ROOT / CASES[request.param], request.param, 0)
    yield request.param, policy.parts
    del policy
    torch.cuda.empty_cache()


def test_all_inputs_reproduce_get_action_exactly(loaded):
    from experiments.robot import robot_utils

    platform, parts = loaded
    instruction = "put both the alphabet soup and the tomato sauce in the basket"
    reference = np.stack(
        robot_utils.get_action(
            parts.cfg,
            parts.vla,
            dict(_obs(platform)),
            instruction,
            processor=parts.processor,
            action_head=parts.action_head,
            proprio_projector=parts.proprio_projector,
            use_film=False,
            use_minivlm=True,
        )
    )
    ours = forward_chunk(parts, dict(_obs(platform)), instruction, Modalities())
    np.testing.assert_array_equal(ours, reference)


@pytest.mark.parametrize("spec", SPECS, ids=repr)
def test_each_withheld_input_runs_and_changes_the_chunk(loaded, spec):
    platform, parts = loaded
    instruction = "open the drawer"
    full = forward_chunk(parts, dict(_obs(platform)), instruction, Modalities())
    chunk = forward_chunk(parts, dict(_obs(platform)), instruction, spec)
    assert chunk.shape == (8, 7) and np.isfinite(chunk).all()
    assert not np.array_equal(chunk, full)


def _training_batch(parts, platform, B=2):
    """A training batch in the layout of upstream's RLDSBatchTransform + collator: prompt,
    64 action tokens and stop, labels on the action tokens and stop only."""
    from experiments.robot import openvla_utils
    from prismatic.vla.constants import ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_TOKENS, STOP_INDEX

    from flower.evaluation.vla_adapter_policy import PROMPT

    device = openvla_utils.DEVICE
    obs = _obs(platform)
    images = openvla_utils.prepare_images_for_vla([obs["full_image"], obs["wrist_image"]], parts.cfg)
    inputs = parts.processor(PROMPT.format("open the drawer"), images[0])
    wrist = parts.processor(PROMPT.format("open the drawer"), images[1])
    prompt_ids = inputs["input_ids"][0]
    tail = torch.tensor([ACTION_TOKEN_BEGIN_IDX + 1 + (i % 7) for i in range(NUM_TOKENS)] + [STOP_INDEX])
    input_ids = torch.cat([prompt_ids, tail]).repeat(B, 1)
    labels = input_ids.clone()
    labels[:, : len(prompt_ids)] = IGNORE_INDEX
    gen = torch.Generator().manual_seed(0)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": torch.ones_like(input_ids).to(device),
        "labels": labels.to(device),
        "pixel_values": torch.cat([inputs["pixel_values"], wrist["pixel_values"]], dim=1).repeat(B, 1, 1, 1),
        # On the device here: in training the DDP-wrapped projector moves its input there.
        "proprio": torch.randn(B, 8, generator=gen).to(device),
        "actions": torch.rand(B, 8, 7, generator=gen),
    }


def test_dropout_training_forward_with_everything_kept_equals_upstream(loaded, monkeypatch):
    """The masked training path (masks injected into upstream run_forward_pass), forced to
    keep every input, gives upstream's loss; a real subset draw changes it."""
    from types import SimpleNamespace

    import flower.models.networks.modality_dropout as md
    from flower.vla_adapter.finetune_dropout import dropout_forward_pass, load_upstream

    platform, parts = loaded
    cwd = os.getcwd()
    upstream = load_upstream()
    os.chdir(cwd)
    batch = _training_batch(parts, platform)
    kwargs = dict(
        vla=parts.vla,
        action_head=SimpleNamespace(module=parts.action_head),
        proprio_projector=parts.proprio_projector,
        batch=batch,
        action_tokenizer=None,
        device_id=0,
        use_l1_regression=True,
        use_proprio=True,
        use_film=False,
        num_patches=512,
        use_pro_version=True,
        cfg=SimpleNamespace(phase="Training"),
    )
    parts.vla.train()
    try:
        with torch.no_grad():
            torch.manual_seed(0)
            reference, _ = upstream.run_forward_pass(**kwargs)
            monkeypatch.setattr(md, "sample_modality_subsets", lambda B, n, device: torch.ones(B, n, dtype=torch.bool))
            torch.manual_seed(0)
            kept, _ = dropout_forward_pass(upstream.run_forward_pass, "subset")(**kwargs)
            monkeypatch.setattr(
                md, "sample_modality_subsets", lambda B, n, device: torch.tensor([[False, True, False, True]] * B)
            )
            torch.manual_seed(0)
            dropped, _ = dropout_forward_pass(upstream.run_forward_pass, "subset")(**kwargs)
    finally:
        parts.vla.eval()
    assert kept.item() == reference.item()
    assert torch.isfinite(dropped) and dropped.item() != reference.item()
