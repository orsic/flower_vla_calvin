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
