"""Tests for flower.models.flower.inference_noise -- pure logic, no model construction.

generators=None is the default (training, CALVIN, validation) path: this must stay
byte-identical to the pre-existing single torch.randn call. Otherwise, each row is
drawn from its own CPU generator, independent of its slot index, the batch width, and
which other rows share the batch -- see the function's own docstring for why this
matters (LIBERO-Plus's cross_task_batching groups unrelated tasks into one batch).
"""
import pytest
import torch

from flower.models.flower import inference_noise


def test_default_path_matches_plain_randn():
    torch.manual_seed(0)
    expected = torch.randn(4, 10, 7)

    torch.manual_seed(0)
    actual = inference_noise(4, 10, 7, device="cpu", generators=None)

    assert torch.equal(actual, expected)


def test_row_is_slot_independent():
    gen_a = torch.Generator().manual_seed(1)
    gen_b = torch.Generator().manual_seed(2)

    forward = inference_noise(2, 3, 4, device="cpu", generators=[gen_a, gen_b])

    gen_a = torch.Generator().manual_seed(1)
    gen_b = torch.Generator().manual_seed(2)
    swapped = inference_noise(2, 3, 4, device="cpu", generators=[gen_b, gen_a])

    assert torch.equal(forward[0], swapped[1])
    assert torch.equal(forward[1], swapped[0])


def test_row_is_batch_width_independent():
    gen_a1 = torch.Generator().manual_seed(1)
    solo = inference_noise(1, 3, 4, device="cpu", generators=[gen_a1])

    gen_a2 = torch.Generator().manual_seed(1)
    gen_b = torch.Generator().manual_seed(2)
    gen_c = torch.Generator().manual_seed(3)
    trio = inference_noise(3, 3, 4, device="cpu", generators=[gen_a2, gen_b, gen_c])

    assert torch.equal(solo[0], trio[0])


def test_stream_advances_and_replays():
    gen = torch.Generator().manual_seed(7)
    first = inference_noise(1, 3, 4, device="cpu", generators=[gen])
    second = inference_noise(1, 3, 4, device="cpu", generators=[gen])
    assert not torch.equal(first, second)

    replay_gen = torch.Generator().manual_seed(7)
    replay_first = inference_noise(1, 3, 4, device="cpu", generators=[replay_gen])
    replay_second = inference_noise(1, 3, 4, device="cpu", generators=[replay_gen])
    assert torch.equal(first, replay_first)
    assert torch.equal(second, replay_second)


def test_generator_count_mismatch_raises():
    gen = torch.Generator().manual_seed(1)
    with pytest.raises(ValueError):
        inference_noise(2, 3, 4, device="cpu", generators=[gen])
