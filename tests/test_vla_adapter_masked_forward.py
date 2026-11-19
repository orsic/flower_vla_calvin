"""flower/vla_adapter/masked_forward.py: per-sample whole-modality removal by key masking."""

import pytest
import torch

from flower.vla_adapter.masked_forward import (
    IGNORE_INDEX,
    drop_masks,
    is_pro_action_head,
    masked_pro_block,
    predict_action_masked,
)

P = 3  # patches per image in these tests


def _batch():
    # Row 0: tok0, 3 prompt tokens, 2 action tokens, stop. Row 1: shorter prompt, 1 pad.
    labels = torch.tensor(
        [
            [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 7, 7, 2],
            [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 7, 7, 2, IGNORE_INDEX],
        ]
    )
    attention = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0]])
    return labels, attention


def test_all_kept_masks_are_the_upstream_masks():
    labels, attention = _batch()
    m = drop_masks(torch.ones(2, 4, dtype=torch.bool), labels, attention, patches_per_image=P)
    expected = torch.cat([attention[:, :1], torch.ones(2, 2 * P, dtype=torch.long), attention[:, 1:]], dim=1).bool()
    assert torch.equal(m.attention, expected)
    assert m.task.all() and m.proprio.all() and m.task.shape == (2, 2 * P)


def test_dropped_camera_masks_its_patches_and_shifted_task_keys():
    labels, attention = _batch()
    keep = torch.tensor([[False, True, True, True], [True, False, True, True]])
    m = drop_masks(keep, labels, attention, patches_per_image=P)
    # attention layout: [tok0, static P, wrist P, rest]
    assert m.attention[0, 1 : 1 + P].tolist() == [False] * P and m.attention[0, 1 + P : 1 + 2 * P].all()
    assert m.attention[1, 1 : 1 + P].all() and m.attention[1, 1 + P : 1 + 2 * P].tolist() == [False] * P
    # task keys = hidden[:, :2P] = [tok0, static 0..P-1, wrist 0..P-2]
    assert m.task[0].tolist() == [True] + [False] * P + [True] * (P - 1)
    assert m.task[1].tolist() == [True] * (1 + P) + [False] * (P - 1)


def test_dropped_language_masks_the_prompt_only():
    labels, attention = _batch()
    keep = torch.tensor([[True, True, False, True], [True, True, False, True]])
    m = drop_masks(keep, labels, attention, patches_per_image=P)
    text = torch.cat([m.attention[:, :1], m.attention[:, 1 + 2 * P :]], dim=1)
    assert text[0].tolist() == [True, False, False, False, True, True, True]
    assert text[1].tolist() == [True, False, False, True, True, True, False]  # pad stays masked


def test_proprio_only_keeps_tok0_actions_and_the_proprio_key():
    labels, attention = _batch()
    keep = torch.tensor([[False, False, False, True]] * 2)
    m = drop_masks(keep, labels, attention, patches_per_image=P)
    assert m.attention[0].tolist() == [True] + [False] * (2 * P) + [False, False, False, True, True, True]
    assert m.task[0].tolist() == [True] + [False] * (2 * P - 1)
    assert m.proprio.all()


def _block():
    from prismatic.models.action_heads import MLPResNetBlock_Pro

    torch.manual_seed(0)
    block = MLPResNetBlock_Pro(dim=16, num_heads=4)
    with torch.no_grad():
        block.gating_factor.fill_(0.7)
    x, h_a, h_t, p = torch.randn(2, 5, 16), torch.randn(2, 4, 16), torch.randn(2, 6, 16), torch.randn(2, 1, 16)
    return block, x, h_a, h_t, p


def test_masked_block_with_all_keys_equals_upstream_block():
    block, x, h_a, h_t, p = _block()
    ours = masked_pro_block(
        block, x, h_a, h_t, p, torch.ones(2, 5, dtype=torch.bool), torch.ones(2, 6, dtype=torch.bool)
    )
    torch.testing.assert_close(ours, block(x, h_a=h_a, h_t=h_t, p=p), rtol=0, atol=0)


def test_masked_keys_have_no_influence():
    block, x, h_a, h_t, p = _block()
    adapter = torch.ones(2, 5, dtype=torch.bool)
    adapter[:, -1] = False  # proprio token
    task = torch.ones(2, 6, dtype=torch.bool)
    task[:, 2:4] = False
    out = masked_pro_block(block, x, h_a, h_t, p, adapter, task)
    h_t2, p2 = h_t.clone(), p * 100
    h_t2[:, 2:4] += 50
    torch.testing.assert_close(masked_pro_block(block, x, h_a, h_t2, p2, adapter, task), out)
    assert not torch.allclose(out, block(x, h_a=h_a, h_t=h_t, p=p))


def test_predict_action_masked_all_kept_equals_upstream_head():
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

    torch.manual_seed(0)
    dim = 16
    head = L1RegressionActionHead(
        input_dim=dim, hidden_dim=dim, action_dim=ACTION_DIM, num_task_tokens=6, use_pro_version=True
    )
    projector = ProprioProjector(llm_dim=dim, proprio_dim=8)
    n_layers = len(head.model.mlp_resnet_blocks) + 1
    hidden = torch.randn(2, n_layers, 6 + 64, dim)
    proprio = torch.randn(2, 8)
    ones_task, ones_p = torch.ones(2, 6, dtype=torch.bool), torch.ones(2, dtype=torch.bool)
    head, projector = head.to(torch.bfloat16), projector.to(torch.bfloat16)
    hidden = hidden.to(torch.bfloat16)
    ours = predict_action_masked(head, hidden, proprio, projector, ones_task, ones_p)
    theirs = head.predict_action(hidden, proprio=proprio, proprio_projector=projector)
    assert ours.shape == (2, NUM_ACTIONS_CHUNK, ACTION_DIM)
    torch.testing.assert_close(ours, theirs, rtol=0, atol=0)


def test_is_pro_action_head():
    assert is_pro_action_head({"model.mlp_resnet_blocks.0.k_task.weight": 0})
    assert not is_pro_action_head({"model.mlp_resnet_blocks.0.q_proj.weight": 0})
