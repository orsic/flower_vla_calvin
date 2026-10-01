"""Regression test for flower-vla-mimicgen:latest shipping MimicGen's non-Python assets.

A plain `pip install git+...` of mimicgen / robosuite-task-zoo copies only .py files
(neither declares package_data), so coffee/coffee_preparation/hammer_cleanup/kitchen/
mug_cleanup fail at env construction with FileNotFoundError on their XML models. Skipped
outside the MimicGen image.
"""

import os

import pytest

mimicgen = pytest.importorskip("mimicgen")
robosuite_task_zoo = pytest.importorskip("robosuite_task_zoo")

MIMICGEN_ASSETS = (
    "models/robosuite/assets/objects/coffee_pod.xml",
    "models/robosuite/assets/objects/drawer.xml",
    "models/robosuite/assets/objects/drawer_long.xml",
)
TASK_ZOO_ASSETS = ("models/kitchen/stove.xml",)


@pytest.mark.parametrize("rel_path", MIMICGEN_ASSETS)
def test_mimicgen_assets_installed(rel_path: str) -> None:
    root = os.path.dirname(mimicgen.__file__)
    assert os.path.isfile(os.path.join(root, rel_path))


@pytest.mark.parametrize("rel_path", TASK_ZOO_ASSETS)
def test_robosuite_task_zoo_assets_installed(rel_path: str) -> None:
    root = os.path.dirname(robosuite_task_zoo.__file__)
    assert os.path.isfile(os.path.join(root, rel_path))
