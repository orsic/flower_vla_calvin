"""Tests for scripts/open_loop_calvin.py's choice and naming of the scored cells.

- A dropout run is scored on every combo of modality_combos (14 with proprio, now with
  distinct labels); a plain run on the inputs it was trained on.
- --masked-withheld adds each trained input withheld: cameras by their mean pixel, language
  as the blank instruction, proprio zeroed.
- Labels and part files of runs without proprio are unchanged, so earlier parts resume.
"""

import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from open_loop_calvin import cell_label, cells, model_label, part_name  # noqa: E402

ALL_ON = {"rgb_static": True, "rgb_gripper": True, "language": True, "proprio": True}


def _cfg(dropout=False, use_proprio=False, modalities=None, keep=0.5, strategy="dirichlet"):
    model = {
        "modality_dropout": dropout,
        "use_proprio": use_proprio,
        "modality_dropout_keep_fraction": keep,
        "modality_dropout_strategy": strategy,
    }
    if modalities is not None:
        model["modalities"] = modalities
    return OmegaConf.create({"model": model})


def test_dropout_run_scores_every_combo_with_distinct_labels():
    no_prop = cells(_cfg(dropout=True), masked_withheld=False)
    assert len(no_prop) == 7 and all(masked is None for _, masked in no_prop)
    with_prop = cells(_cfg(dropout=True, use_proprio=True), masked_withheld=False)
    labels = {cell_label(c, m, True) for c, m in with_prop}
    assert len(with_prop) == 14 and len(labels) == 14


def test_plain_run_scores_its_trained_inputs():
    assert cells(_cfg(), masked_withheld=False) == [(ALL_ON, None)]
    subset = {"rgb_static": True, "rgb_gripper": False, "language": True, "proprio": True}
    [(combo, masked)] = cells(_cfg(use_proprio=True, modalities=subset), masked_withheld=False)
    assert combo == subset and masked is None
    assert cell_label(combo, masked, True) == "static+lang+proprio"


def test_masked_withheld_adds_each_trained_input_withheld():
    got = cells(_cfg(use_proprio=True), masked_withheld=True)
    assert got == [
        (ALL_ON, None),
        (ALL_ON, "static"),
        (ALL_ON, "wrist"),
        (ALL_ON, "lang"),
        ({**ALL_ON, "proprio": False}, None),
    ]
    assert [cell_label(c, m, True) for c, m in got] == [
        "static+wrist+lang+proprio",
        "wrist+lang+proprio (static masked)",
        "static+lang+proprio (wrist masked)",
        "static+wrist+proprio (lang masked)",
        "static+wrist+lang",
    ]
    assert part_name(ALL_ON, "static", True) == "wrist+lang+proprio.masked_static.csv.gz"
    # Without proprio there is nothing to zero, and a fixed subset masks only its own inputs.
    assert len(cells(_cfg(), masked_withheld=True)) == 4
    subset = {"rgb_static": True, "rgb_gripper": False, "language": True, "proprio": True}
    masks = [m for _, m in cells(_cfg(use_proprio=True, modalities=subset), masked_withheld=True)]
    assert masks == [None, "static", "lang", None]


def test_labels_of_runs_without_proprio_are_unchanged():
    assert cell_label(ALL_ON, None, False) == "static+wrist+lang"
    assert part_name(ALL_ON, None, False) == "static+wrist+lang.csv.gz"
    assert part_name({**ALL_ON, "rgb_gripper": False}, None, False) == "static+lang.csv.gz"


def test_model_label():
    assert model_label(_cfg()) == "no dropout"
    assert model_label(_cfg(dropout=True, keep=0.5)) == "dirichlet keep0.5"
    assert model_label(_cfg(use_proprio=True)) == "no dropout + proprio"
    assert model_label(_cfg(dropout=True, use_proprio=True, strategy="subset")) == "subset keep0.5 + proprio"
    subset = {"rgb_static": True, "rgb_gripper": False, "language": True, "proprio": True}
    assert model_label(_cfg(use_proprio=True, modalities=subset)) == "no dropout + proprio (trained without wrist)"
