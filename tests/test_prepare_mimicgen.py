"""Tests for scripts/prepare_mimicgen.py's per-dataset render driver.

No robomimic/mimicgen involved: subprocess.run is monkeypatched out, so these only
exercise the skip/error/cleanup logic around the (mocked) render call.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import prepare_mimicgen  # noqa: E402


def test_render_one_skips_when_output_already_exists(tmp_path, monkeypatch):
    data_dir = tmp_path
    (data_dir / "square_d0.hdf5").write_bytes(b"fake")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called when output already exists")

    monkeypatch.setattr(prepare_mimicgen.subprocess, "run", fail_if_called)

    prepare_mimicgen.render_one("square_d0", str(data_dir), n_demo=100)  # must not raise


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
        (data_dir / "square_d0.hdf5").write_bytes(b"fake rendered")

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


def test_render_one_keeps_source_when_env_var_set(tmp_path, monkeypatch):
    data_dir = tmp_path
    source_dir = data_dir / "source" / "core"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "square_d0.hdf5"
    source_path.write_bytes(b"fake source")

    def fake_run(cmd, check):
        (data_dir / "square_d0.hdf5").write_bytes(b"fake rendered")

    monkeypatch.setattr(prepare_mimicgen.subprocess, "run", fake_run)
    monkeypatch.setenv("KEEP_MIMICGEN_SOURCE", "1")

    prepare_mimicgen.render_one("square_d0", str(data_dir), n_demo=50)

    assert source_path.exists()


def test_main_rejects_unknown_dataset(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["prepare_mimicgen.py", "not_a_real_dataset"])
    with pytest.raises(ValueError, match="unknown dataset"):
        prepare_mimicgen.main()
