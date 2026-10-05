"""Tests for scripts/prepare_mimicgen.py's per-dataset render driver.

No robomimic/mimicgen involved: subprocess.run is monkeypatched out, so these only
exercise the skip/error/cleanup logic around the (mocked) render call.
"""
import json
import subprocess
import sys
from pathlib import Path

import h5py
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import prepare_mimicgen  # noqa: E402


def _output_name(cmd) -> Path:
    """The --output_name the (mocked) converter was told to write to."""
    argv_line = next(line for line in cmd[2].splitlines() if line.startswith("sys.argv"))
    argv = json.loads(argv_line.split("=", 1)[1].strip())
    return Path(argv[argv.index("--output_name") + 1])


def _write_rendered(path: Path, heights, widths) -> None:
    """A rendered output as far as render_one's up-to-date check is concerned: just the
    env_args robomimic records, with the camera sizes it was rendered at."""
    env_kwargs = {
        "camera_names": list(prepare_mimicgen.CAMERA_SIZES),
        "camera_heights": heights,
        "camera_widths": widths,
    }
    with h5py.File(path, "w") as f:
        f.create_group("data").attrs["env_args"] = json.dumps({"env_kwargs": env_kwargs})


def test_render_one_skips_when_output_already_at_current_camera_sizes(tmp_path, monkeypatch):
    sizes = list(prepare_mimicgen.CAMERA_SIZES.values())
    _write_rendered(tmp_path / "square_d0.hdf5", sizes, sizes)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called when output is up to date")

    monkeypatch.setattr(prepare_mimicgen.subprocess, "run", fail_if_called)

    prepare_mimicgen.render_one("square_d0", str(tmp_path), n_demo=100)  # must not raise


def test_render_one_rerenders_output_at_stale_camera_sizes(tmp_path, monkeypatch):
    """An output from the old 128x128 recipe is re-rendered and replaced in place."""
    _make_source(tmp_path)
    _write_rendered(tmp_path / "square_d0.hdf5", 128, 128)
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        _output_name(cmd).write_bytes(b"fake rendered")

    monkeypatch.setattr(prepare_mimicgen.subprocess, "run", fake_run)

    prepare_mimicgen.render_one("square_d0", str(tmp_path), n_demo=100)

    assert len(calls) == 1
    assert (tmp_path / "square_d0.hdf5").read_bytes() == b"fake rendered"


def test_render_one_renders_per_camera_sizes_without_next_obs(tmp_path, monkeypatch):
    _make_source(tmp_path)
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        _output_name(cmd).write_bytes(b"fake rendered")

    monkeypatch.setattr(prepare_mimicgen.subprocess, "run", fake_run)

    prepare_mimicgen.render_one("square_d0", str(tmp_path), n_demo=100)

    script = calls[0][2]
    argv = json.loads(next(l for l in script.splitlines() if l.startswith("sys.argv")).split("=", 1)[1])
    assert "--exclude-next-obs" in argv  # training never reads next_obs (load_next_obs=False)
    assert prepare_mimicgen.CAMERA_SIZES == {"agentview": 200, "robot0_eye_in_hand": 84}  # CALVIN's native sizes
    assert json.dumps(prepare_mimicgen.CAMERA_SIZES) in script
    assert "create_env_for_data_processing" in script


def test_render_one_raises_when_source_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="download-mimicgen"):
        prepare_mimicgen.render_one("square_d0", str(tmp_path), n_demo=100)


def test_render_one_invokes_converter_and_deletes_source(tmp_path, monkeypatch):
    data_dir = tmp_path
    source_dir = data_dir / "source" / "core"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "square_d0.hdf5"
    source_path.write_bytes(b"fake source")

    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        assert check is True
        # Simulate the converter producing the output file.
        _output_name(cmd).write_bytes(b"fake rendered")

    monkeypatch.setattr(prepare_mimicgen.subprocess, "run", fake_run)
    monkeypatch.delenv("KEEP_MIMICGEN_SOURCE", raising=False)

    prepare_mimicgen.render_one("square_d0", str(data_dir), n_demo=50)

    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == sys.executable
    assert cmd[1] == "-c"
    script = cmd[2]
    assert "import mimicgen" in script
    assert "robomimic.scripts.dataset_states_to_obs" in script
    # argv embedded as a JSON list literal -- pull it back out to check the actual flags.
    argv_line = next(line for line in script.splitlines() if line.startswith("sys.argv"))
    argv = json.loads(argv_line.split("=", 1)[1].strip())
    assert "--n" in argv and argv[argv.index("--n") + 1] == "50"
    assert "--dataset" in argv and argv[argv.index("--dataset") + 1] == str(source_path)
    assert "agentview" in argv and "robot0_eye_in_hand" in argv

    assert not source_path.exists()  # deleted after a successful render
    assert (data_dir / "square_d0.hdf5").read_bytes() == b"fake rendered"
    assert not (data_dir / "square_d0.hdf5.tmp").exists()


def _make_source(data_dir: Path) -> Path:
    source_dir = data_dir / "source" / "core"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "square_d0.hdf5"
    source_path.write_bytes(b"fake source")
    return source_path


def test_render_one_failed_render_leaves_no_output(tmp_path, monkeypatch):
    """A render that dies midway (crash, container kill) must not leave a file at the
    final path -- render_one would otherwise skip it as done on the next run."""
    source_path = _make_source(tmp_path)

    def dying_run(cmd, check):
        _output_name(cmd).write_bytes(b"half written")
        raise subprocess.CalledProcessError(-9, cmd)

    monkeypatch.setattr(prepare_mimicgen.subprocess, "run", dying_run)

    with pytest.raises(subprocess.CalledProcessError):
        prepare_mimicgen.render_one("square_d0", str(tmp_path), n_demo=100)

    assert not (tmp_path / "square_d0.hdf5").exists()
    assert source_path.exists()


def test_render_one_rerenders_over_stale_tmp(tmp_path, monkeypatch):
    _make_source(tmp_path)
    (tmp_path / "square_d0.hdf5.tmp").write_bytes(b"stale partial")
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        assert not _output_name(cmd).exists()  # stale partial removed before rendering
        _output_name(cmd).write_bytes(b"fake rendered")

    monkeypatch.setattr(prepare_mimicgen.subprocess, "run", fake_run)

    prepare_mimicgen.render_one("square_d0", str(tmp_path), n_demo=100)

    assert len(calls) == 1
    assert (tmp_path / "square_d0.hdf5").read_bytes() == b"fake rendered"


def test_render_one_keeps_source_when_env_var_set(tmp_path, monkeypatch):
    data_dir = tmp_path
    source_dir = data_dir / "source" / "core"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "square_d0.hdf5"
    source_path.write_bytes(b"fake source")

    def fake_run(cmd, check):
        _output_name(cmd).write_bytes(b"fake rendered")

    monkeypatch.setattr(prepare_mimicgen.subprocess, "run", fake_run)
    monkeypatch.setenv("KEEP_MIMICGEN_SOURCE", "1")

    prepare_mimicgen.render_one("square_d0", str(data_dir), n_demo=50)

    assert source_path.exists()


def test_main_rejects_unknown_dataset(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["prepare_mimicgen.py", "not_a_real_dataset"])
    with pytest.raises(ValueError, match="unknown dataset"):
        prepare_mimicgen.main()
