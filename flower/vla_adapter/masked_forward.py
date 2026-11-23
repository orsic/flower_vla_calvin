"""Whole-modality removal for VLA-Adapter by key masking, shared by dropout training
(flower/vla_adapter/finetune_dropout.py) and evaluation (flower/evaluation/vla_adapter_policy.py).

FlowerVLA removes a dropped modality's tokens and keeps every other token at its
full-layout position (flower/models/flower.py: compact_layout / position_correct). In
VLA-Adapter's causal LLM the same is a key mask at unchanged positions: a dropped camera's
patch tokens, or the prompt tokens (all but the first, which anchors the layout like
FlowerVLA's <Flow> token), get attention-mask 0, and Qwen2's default position ids keep the
full layout. The action head cross-attends to the patch states ("task" keys) and to the
action-query states plus the proprio token ("adapter" keys); a masked copy of its Pro block
leaves out the dropped patch states and the dropped proprio token there, keeping its RoPE
positions over the full layout too. With nothing dropped every mask is all-True and the
upstream computation is unchanged.

Per-sample keep flags are (static, wrist, language, proprio), the column order of
flower.models.networks.modality_dropout.sample_modality_subsets.
"""

import math
from contextlib import contextmanager
from typing import NamedTuple, Optional

import torch

IGNORE_INDEX = -100  # prismatic.vla.constants.IGNORE_INDEX
_MISSING = object()


class DropMasks(NamedTuple):
    """attention: (B, 1 + n_patches + S - 1) LLM key mask over [tok0, patches, rest];
    task: (B, n_patches) action-head task-key mask, aligned with the upstream task states
    hidden[:, :n_patches] (tok0, then the patches shifted by one); proprio: (B,)."""

    attention: torch.Tensor
    task: torch.Tensor
    proprio: torch.Tensor


def drop_masks(
    keep: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor, patches_per_image: int = 256
) -> DropMasks:
    """Masks for per-sample keep flags (B, 4) = (static, wrist, language, proprio), given the
    text-side labels and attention mask (B, S) the upstream forward sees. The prompt is
    every token after the first up to the first labelled (action) token."""
    keep = keep.bool().to(labels.device)
    batch, seq = labels.shape
    positions = torch.arange(seq, device=labels.device)
    first_action = (labels != IGNORE_INDEX).float().argmax(dim=1, keepdim=True)
    prompt = (positions >= 1) & (positions < first_action)
    text = attention_mask.bool() & ~(prompt & ~keep[:, 2:3])
    patches = torch.cat(
        [keep[:, 0:1].expand(batch, patches_per_image), keep[:, 1:2].expand(batch, patches_per_image)], dim=1
    )
    attention = torch.cat([text[:, :1], patches, text[:, 1:]], dim=1)
    task = torch.cat([torch.ones_like(patches[:, :1]), patches[:, :-1]], dim=1)
    return DropMasks(attention, task, keep[:, 3])


def masked_pro_block(block, x, h_a, h_t, p, adapter_mask: torch.Tensor, task_mask: torch.Tensor):
    """MLPResNetBlock_Pro.forward with key masks: adapter_mask (B, K_a) over [h_a, p] and
    task_mask (B, K_t) over h_t; masked keys get -inf before the softmax. Otherwise a
    line-for-line copy of the upstream block (prismatic/models/action_heads.py)."""
    from prismatic.models.action_heads import apply_rope

    ratio_g = torch.tanh(block.gating_factor)
    h_adapter = torch.cat((h_a, p), dim=1)
    h_task = h_t
    B, T, C = x.shape
    K_a = h_adapter.size(1)
    K_t = h_task.size(1)

    q_1 = block.q_proj(x)
    k_tokens = block.k_self(x)
    v_tokens = block.v_self(x)
    k_adapter = block.k_adapter(h_adapter)
    v_adapter = block.v_adapter(h_adapter)
    k_task = block.k_task(h_task)
    v_task = block.v_task(h_task)

    def reshape_heads(t, B, L):
        return t.view(B, L, block.num_heads, block.head_dim).transpose(1, 2)

    q_1 = reshape_heads(q_1, B, T)
    k_tokens, v_tokens = reshape_heads(k_tokens, B, T), reshape_heads(v_tokens, B, T)
    k_adapter, v_adapter = reshape_heads(k_adapter, B, K_a), reshape_heads(v_adapter, B, K_a)
    k_task, v_task = reshape_heads(k_task, B, K_t), reshape_heads(v_task, B, K_t)

    cos_main, sin_main = block.rope(seq_len=T, device=x.device, dtype=x.dtype)
    q_1, k_tokens = apply_rope(q_1, k_tokens, cos_main, sin_main)
    cos_a, sin_a = block.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
    _, k_adapter = apply_rope(k_adapter, k_adapter, cos_a, sin_a)
    cos_t, sin_t = block.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
    _, k_task = apply_rope(k_task, k_task, cos_t, sin_t)

    scores_adapter = torch.matmul(q_1, k_adapter.transpose(-2, -1))
    scores_task = torch.matmul(q_1, k_task.transpose(-2, -1)) * ratio_g
    scores_adapter = scores_adapter.masked_fill(~adapter_mask[:, None, None, :], float("-inf"))
    scores_task = scores_task.masked_fill(~task_mask[:, None, None, :], float("-inf"))
    attn_scores = [torch.matmul(q_1, k_tokens.transpose(-2, -1)), scores_adapter, scores_task]
    attn_scores = torch.cat(attn_scores, dim=-1) / math.sqrt(block.head_dim)
    attn_weights = torch.softmax(attn_scores, dim=-1)

    v_combined = torch.cat([v_tokens, v_adapter, v_task], dim=2)
    output = torch.matmul(attn_weights, v_combined)
    output = output.transpose(1, 2).contiguous().view(B, T, C)
    output = block.o_proj(output)
    return block.ffn(output + x)


def predict_action_masked(
    action_head,
    actions_hidden_states: torch.Tensor,
    proprio: torch.Tensor,
    proprio_projector,
    task_mask: torch.Tensor,
    proprio_mask: torch.Tensor,
    phase: str = "Inference",
) -> torch.Tensor:
    """L1RegressionActionHead.predict_action (+ MLPResNet.forward) with the task keys and
    the proprio key masked per sample. Needs the Pro head."""
    from prismatic.models.action_heads import MLPResNetBlock_Pro, learnable_random_perturbations
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK

    batch_size = actions_hidden_states.shape[0]
    device = actions_hidden_states.device
    proprio = proprio.reshape(batch_size, -1).to(torch.bfloat16)
    proprio_features = proprio_projector(proprio).unsqueeze(dim=1)

    task_hidden_states = actions_hidden_states[:, :, : action_head.num_task_tokens, :]
    actions_hidden_states = actions_hidden_states[:, :, action_head.num_task_tokens :, :]
    cond = torch.zeros(
        (batch_size, action_head.action_dim * NUM_ACTIONS_CHUNK, action_head.hidden_dim),
        device=device,
        dtype=actions_hidden_states.dtype,
    ).detach()
    x = cond.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
    if phase == "Training":
        _, seq_len, dim = x.shape
        x = x + learnable_random_perturbations(seq_len, dim, device=x.device, dtype=x.dtype)

    n_adapter = actions_hidden_states.shape[2] + 1
    adapter_mask = torch.ones((batch_size, n_adapter), dtype=torch.bool, device=device)
    adapter_mask[:, -1] = proprio_mask.to(device)
    task_mask = task_mask.to(device)

    model = action_head.model
    x = model.relu(model.fc1(model.layer_norm1(x)))
    for i, block in enumerate(model.mlp_resnet_blocks):
        if not isinstance(block, MLPResNetBlock_Pro):
            raise NotImplementedError("Masked removal needs VLA-Adapter's Pro action head")
        x = masked_pro_block(
            block,
            x,
            h_a=actions_hidden_states[:, i + 1, :],
            h_t=task_hidden_states[:, i + 1, :],
            p=proprio_features,
            adapter_mask=adapter_mask,
            task_mask=task_mask,
        )
    return model.fc2(model.layer_norm2(x))


@contextmanager
def injected_masks(model, action_head, masks: DropMasks):
    """Within the block, `model`'s multimodal attention mask is masks.attention and
    `action_head.predict_action` masks the dropped task and proprio keys, so upstream code
    (finetune.run_forward_pass, predict_action) runs unchanged on top. `model` is the
    OpenVLAForActionPrediction itself (not its DDP / PEFT wrapper)."""

    def build_multimodal_attention(input_embeddings, projected_patch_embeddings, attention_mask):
        embeddings, attention = type(model)._build_multimodal_attention(
            model, input_embeddings, projected_patch_embeddings, attention_mask
        )
        if attention.shape != masks.attention.shape:
            raise ValueError(f"attention mask {tuple(masks.attention.shape)} vs sequence {tuple(attention.shape)}")
        return embeddings, masks.attention.to(device=attention.device, dtype=attention.dtype)

    def predict_action(actions_hidden_states, proprio=None, proprio_projector=None, phase="Inference"):
        return predict_action_masked(
            action_head, actions_hidden_states, proprio, proprio_projector, masks.task, masks.proprio, phase
        )

    patches = [
        (model, "_build_multimodal_attention", build_multimodal_attention),
        (action_head, "predict_action", predict_action),
    ]
    saved = [(obj, name, vars(obj).get(name, _MISSING)) for obj, name, _ in patches]
    for obj, name, fn in patches:
        setattr(obj, name, fn)
    try:
        yield
    finally:
        for obj, name, previous in saved:
            if previous is _MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, previous)


def unwrap(module):
    """The OpenVLAForActionPrediction under DDP (.module) and PEFT (.base_model.model)."""
    if isinstance(module, torch.nn.parallel.DistributedDataParallel):
        module = module.module
    if type(module).__name__.startswith("Peft"):
        module = module.base_model.model
    return module


def is_pro_action_head(state_dict: dict) -> bool:
    """Whether an action-head checkpoint holds Pro blocks (separate task/adapter keys)."""
    return any(key.endswith("k_task.weight") for key in state_dict)


def keep_flags(static: bool, wrist: bool, lang: bool, proprio: bool, device: Optional[torch.device] = None):
    return torch.tensor([[static, wrist, lang, proprio]], dtype=torch.bool, device=device)
