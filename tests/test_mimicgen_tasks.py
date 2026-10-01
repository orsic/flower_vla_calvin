"""Tests for flower.datasets.mimicgen_tasks, the MimicGen task registry."""

import pytest

from flower.datasets import mimicgen_tasks as tasks


def test_26_datasets():
    assert len(tasks.CORE_DATASETS) == 26
    assert len(set(tasks.CORE_DATASETS)) == 26


@pytest.mark.parametrize("dataset", tasks.CORE_DATASETS)
def test_every_dataset_resolves(dataset):
    fam = tasks.family(dataset)
    assert fam in tasks.LANGUAGE
    instruction = tasks.language(dataset)
    assert isinstance(instruction, str) and instruction
    assert tasks.max_steps(dataset) > 0


def test_family_strips_difficulty_suffix():
    assert tasks.family("square_d0") == "square"
    assert tasks.family("three_piece_assembly_d2") == "three_piece_assembly"
    # A family name is its own fixed point (no suffix to strip).
    assert tasks.family("square") == "square"


def test_language_shared_across_difficulty_variants():
    assert tasks.language("square_d0") == tasks.language("square_d2")


def test_max_steps_per_family_not_default():
    # square is registered with a non-default max_steps.
    assert tasks.max_steps("square_d0") == 500
    assert tasks.max_steps("square_d0") != tasks.DEFAULT_MAX_STEPS


# Upstream MimicGen's rollout horizons (experiment.rollout.horizon in
# mimicgen/scripts/generate_core_training_configs.py). A shorter cap than the demos
# themselves silently zeroes a task: coffee_preparation's old 500 was below every one of
# its 591-760-step demos.
UPSTREAM_HORIZON = {
    "stack": 400,
    "stack_three": 400,
    "square": 400,
    "threading": 400,
    "coffee": 400,
    "three_piece_assembly": 500,
    "nut_assembly": 500,
    "mug_cleanup": 500,
    "hammer_cleanup": 500,
    "coffee_preparation": 800,
    "kitchen": 800,
    "pick_place": 1000,
}


@pytest.mark.parametrize("dataset", tasks.CORE_DATASETS)
def test_max_steps_is_upstream_horizon_plus_25_percent(dataset):
    # coffee_preparation stays at upstream's 800 (already above its 760-step longest demo);
    # every other family gets 25% headroom over upstream.
    fam = tasks.family(dataset)
    expected = UPSTREAM_HORIZON[fam] if fam == "coffee_preparation" else UPSTREAM_HORIZON[fam] * 5 // 4
    assert tasks.max_steps(dataset) == expected


def test_max_steps_falls_back_to_default_for_unregistered_family():
    assert tasks.max_steps("totally_unknown_task_d0") == tasks.DEFAULT_MAX_STEPS


def test_language_raises_on_unknown_family():
    with pytest.raises(ValueError):
        tasks.language("totally_unknown_task_d0")


def test_env_name():
    assert tasks.env_name("square_d0") == "Square_D0"
