"""Registry of stable facts about the MimicGen `core` datasets.

These are properties of the MimicGen benchmark itself (which env a dataset uses, how
long a rollout gets, what to say to the language encoder), not experiment choices, so
they live in Python rather than being duplicated across Hydra configs. Mirrors the
task registry in https://github.com/zheyu-zhuang/visuomotor-stack
(`visuomotor/data/mimicgen/tasks.py` + `visuomotor/config/tasks.py`), extended with
`coffee`, `stack`, `hammer_cleanup`, and `kitchen` language descriptions so every
dataset in the 26-file `core` release has an instruction.
"""

import re

# The 26 MimicGen `core` datasets (amandlek/mimicgen_datasets on HuggingFace),
# 12 task families across difficulty variants d0/d1/d2.
CORE_DATASETS = (
    "coffee_d0",
    "coffee_d1",
    "coffee_d2",
    "coffee_preparation_d0",
    "coffee_preparation_d1",
    "hammer_cleanup_d0",
    "hammer_cleanup_d1",
    "kitchen_d0",
    "kitchen_d1",
    "mug_cleanup_d0",
    "mug_cleanup_d1",
    "nut_assembly_d0",
    "pick_place_d0",
    "square_d0",
    "square_d1",
    "square_d2",
    "stack_d0",
    "stack_d1",
    "stack_three_d0",
    "stack_three_d1",
    "threading_d0",
    "threading_d1",
    "threading_d2",
    "three_piece_assembly_d0",
    "three_piece_assembly_d1",
    "three_piece_assembly_d2",
)

# One instruction per task family; every difficulty variant of a family shares it.
LANGUAGE = {
    "coffee": "make coffee using the coffee machine and a pod",
    "coffee_preparation": "make coffee using the coffee machine and a pod",
    "hammer_cleanup": "put the hammer in the drawer and close it",
    "kitchen": "cook the food on the stove and serve it",
    "mug_cleanup": "store the mug inside the drawer",
    "nut_assembly": "assemble both square and round nuts onto their pegs",
    "pick_place": "collect all objects and place them into the container",
    "square": "insert the square nut onto the square peg",
    "stack": "stack the blocks into a tower",
    "stack_three": "stack three blocks into a vertical tower",
    "threading": "thread the needle through the eye",
    "three_piece_assembly": "assemble the three toy pieces together",
}

# Per-family max rollout length (from visuomotor-stack's MAX_STEPS registry).
DEFAULT_MAX_STEPS = 800
_FAMILY_MAX_STEPS = {
    "square": 400,
    "stack": 400,
    "stack_three": 400,
    "threading": 400,
    "coffee": 400,
    "coffee_preparation": 500,
    "three_piece_assembly": 500,
    "hammer_cleanup": 500,
    "mug_cleanup": 500,
    "nut_assembly": 500,
    "kitchen": 800,
    "pick_place": 1000,
}


def family(dataset: str) -> str:
    """Strip the trailing difficulty suffix: 'square_d0' -> 'square'."""
    return re.sub(r"_d\d+$", "", str(dataset))


def language(dataset: str) -> str:
    """Instruction for a dataset's task family. Raises on an unregistered family."""
    fam = family(dataset)
    try:
        return LANGUAGE[fam]
    except KeyError as error:
        raise ValueError(f"unknown MimicGen task family: {fam!r}") from error


def max_steps(dataset: str) -> int:
    """Max rollout length for a dataset, falling back to DEFAULT_MAX_STEPS."""
    return _FAMILY_MAX_STEPS.get(family(dataset), DEFAULT_MAX_STEPS)


def env_name(dataset: str) -> str:
    """Robosuite/MimicGen env class name for a dataset: 'square_d0' -> 'Square_D0'."""
    return "_".join(part.capitalize() for part in str(dataset).split("_"))
