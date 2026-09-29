"""Tests for scripts/mimicgen_pipeline.py -- deciding the single MimicGen evaluation a
training run needs, and resolving its result.csv for upload.

Pure logic / filesystem tests, no MuJoCo, no model, no wandb network calls. Mirrors
tests/test_eval_pipeline.py's fixture pattern.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import mimicgen_pipeline  # noqa: E402
from mimicgen_pipeline import _resolve, plan_lines  # noqa: E402

from flower.evaluation.eval_records import result_dir, write_csv  # noqa: E402


def _write_train_cfg(train_folder: Path, *, seed=42, project="multimodal_florence", entity="some-entity"):
    hydra_dir = train_folder / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    (hydra_dir / "config.yaml").write_text(
        f"""
benchmark_name: mimicgen_core
seed: {seed}
model:
  use_proprio: false
logger:
  project: {project}
  entity: {entity}
"""
    )


def test_resolve_default_checkpoint(tmp_path):
    train_folder = tmp_path / "mimicgen" / "run1"
    _write_train_cfg(train_folder, seed=42)

    resolved_train_folder, checkpoint = _resolve(str(train_folder), [])

    assert resolved_train_folder == str(train_folder)
    assert checkpoint == str(train_folder / "seed_42" / "saved_models" / "last.ckpt")


def test_resolve_honors_checkpoint_override(tmp_path):
    train_folder = tmp_path / "mimicgen" / "run1"
    _write_train_cfg(train_folder)

    _, checkpoint = _resolve(str(train_folder), ["checkpoint=/custom/last.ckpt"])

    assert checkpoint == "/custom/last.ckpt"


def test_plan_lines_single_eval_mimicgen_line(tmp_path):
    train_folder = tmp_path / "mimicgen" / "run1"
    _write_train_cfg(train_folder)

    plan = plan_lines(str(train_folder), [], reeval=False)

    assert len(plan) == 1
    service, overrides = plan[0]
    assert service == "eval-mimicgen"
    assert f"train_folder={train_folder}" in overrides
    assert any(o.startswith("checkpoint=") for o in overrides)
    assert "reeval=True" not in overrides


def test_plan_lines_reeval_adds_flag(tmp_path):
    train_folder = tmp_path / "mimicgen" / "run1"
    _write_train_cfg(train_folder)

    plan = plan_lines(str(train_folder), [], reeval=True)

    _, overrides = plan[0]
    assert "reeval=True" in overrides


def test_upload_reports_nothing_when_no_result_csv(tmp_path, capsys):
    train_folder = tmp_path / "mimicgen" / "run1"
    _write_train_cfg(train_folder)

    mimicgen_pipeline.upload(str(train_folder), [])

    assert "Nothing to upload" in capsys.readouterr().out


def test_upload_artifact_name_and_run_id(tmp_path, monkeypatch):
    """upload() must call wandb.init with the same run id LIBERO's own pipeline
    reconstructs, and attach the result.csv as a single 'mimicgen.csv' member --
    without actually reaching the network (wandb is monkeypatched out)."""
    train_folder = tmp_path / "some_group" / "run1"
    _write_train_cfg(train_folder)
    checkpoint = str(train_folder / "seed_42" / "saved_models" / "last.ckpt")
    csv_path = result_dir(str(train_folder), checkpoint, "mimicgen", "core") / "result.csv"
    write_csv(csv_path, [{"task_name": "square_d0", "success": 1}])

    calls = {}

    class _FakeArtifact:
        def __init__(self, name, type):
            calls["artifact_name"] = name
            calls["artifact_type"] = type
            self.files = []

        def add_file(self, path, name):
            self.files.append((path, name))

    class _FakeRun:
        def log_artifact(self, artifact):
            calls["logged_files"] = artifact.files

        def finish(self):
            calls["finished"] = True

    def fake_init(project, entity, id, resume):
        calls["init_kwargs"] = dict(project=project, entity=entity, id=id, resume=resume)
        return _FakeRun()

    monkeypatch.setattr(mimicgen_pipeline.wandb, "init", fake_init)
    monkeypatch.setattr(mimicgen_pipeline.wandb, "Artifact", _FakeArtifact)

    mimicgen_pipeline.upload(str(train_folder), [])

    assert calls["init_kwargs"]["id"] == "some_group_run1"
    assert calls["init_kwargs"]["project"] == "multimodal_florence"
    assert calls["artifact_name"] == "eval-some_group_run1"
    assert calls["artifact_type"] == "evaluation"
    assert calls["logged_files"] == [(str(csv_path), "mimicgen.csv")]
    assert calls["finished"] is True
