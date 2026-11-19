"""scripts/vla_adapter_pipeline.py: VLA-Adapter's withheld-input evals planned with the
FlowerVLA pipeline helpers and collected under FlowerVLA's artifact member names."""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import vla_adapter_pipeline as vap  # noqa: E402

FLAGS = ("use_rgb_static", "use_rgb_gripper", "use_language", "use_proprio")


def _write_rows(path, combos):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[*FLAGS, "task_category_filter"])
        w.writeheader()
        for combo in combos:
            w.writerow({**dict(zip(FLAGS, combo)), "task_category_filter": ""})


def _modalities(line):
    return {tok.split("=")[0].split(".")[1]: tok.split("=")[1] for tok in line if tok.startswith("eval_modalities.")}


@pytest.mark.parametrize("benchmark", ["libero", "calvin"])
def test_plan_all_inputs_each_removed_then_each_masked(tmp_path, benchmark):
    lines = vap.plan_lines(benchmark, str(tmp_path), resume=False)
    assert len(lines) == 9
    script = vap.BENCHMARKS[benchmark]["script"]
    assert all(s == script for s, _ in lines)
    ckpt = tmp_path / vap.BENCHMARKS[benchmark]["checkpoint"]
    for _, line in lines:
        assert line[:3] == ["policy=vla_adapter", f"train_folder={ckpt}", f"checkpoint={ckpt}"]
    mods = [_modalities(line) for _, line in lines]
    assert mods[0] == {"rgb_static": "True", "rgb_gripper": "True", "language": "True", "proprio": "True"}
    removed = [{k for k, v in m.items() if v == "False"} for m in mods[1:5]]
    assert removed == [{"rgb_static"}, {"rgb_gripper"}, {"language"}, {"proprio"}]
    masked = [{k for k, v in m.items() if v == "False"} for m in mods[5:]]
    assert masked == [{"rgb_static_pixels"}, {"rgb_gripper_pixels"}, {"instruction"}, {"proprio_state"}]


def test_libero_lines_pair_with_flowervla_episodes():
    _, line = vap.plan_lines("libero", "/x", resume=False)[0]
    assert "n_eval=20" in line and "env_img_res=256" in line
    assert not any(tok.startswith(("num_steps_wait", "wait_gripper")) for tok in line)  # harness warm-up


def test_resume_skips_combos_and_masked_variants_already_recorded(tmp_path):
    csv_path = vap.result_csv("calvin", str(tmp_path))
    _write_rows(csv_path, [("1", "1", "1", "1"), ("0", "1", "1", "1")])
    _write_rows(csv_path.parent.parent / "masked_wrist" / csv_path.parent.name / "result.csv", [("1", "1", "1", "1")])
    lines = vap.plan_lines("calvin", str(tmp_path), resume=True)
    mods = [{k for k, v in _modalities(line).items() if v == "False"} for _, line in lines]
    assert set() not in mods and {"rgb_static"} not in mods and {"rgb_gripper_pixels"} not in mods
    assert len(lines) == 6


def test_collect_uses_flowervla_member_names(tmp_path):
    csv_path = vap.result_csv("libero", str(tmp_path))
    _write_rows(csv_path, [("1", "1", "1", "1")])
    _write_rows(
        csv_path.parent.parent / "blank_instruction" / csv_path.parent.name / "result.csv", [("1", "1", "1", "1")]
    )
    copies = vap.collect("libero", str(tmp_path))
    assert set(copies) == {"libero_orig.csv", "blank_instruction/libero_orig.csv"}
    art = tmp_path / "LIBERO-Long-Pro" / "artifact"
    assert (art / "libero_orig.csv").read_text() == csv_path.read_text()
    assert (art / "blank_instruction" / "libero_orig.csv").exists()
