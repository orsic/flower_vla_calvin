"""Tests for scripts/mimicgen_pipeline.py -- deciding the MimicGen evaluation a
training run needs (one line per modality combo for a dropout run), and resolving
its result.csv for upload.

Pure logic / filesystem tests, no MuJoCo, no model, no wandb network calls. Mirrors
tests/test_eval_pipeline.py's fixture pattern.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import mimicgen_pipeline  # noqa: E402
from mimicgen_pipeline import _resolve, plan_lines  # noqa: E402

from flower.evaluation.eval_records import result_dir, write_csv  # noqa: E402


def _write_train_cfg(
    train_folder: Path,
    *,
    seed=42,
    project="multimodal_florence",
    entity="some-entity",
    modality_dropout=False,
    use_proprio=False,
):
    hydra_dir = train_folder / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    (hydra_dir / "config.yaml").write_text(
        f"""
benchmark_name: mimicgen_core
seed: {seed}
model:
  use_proprio: {str(use_proprio).lower()}
  modality_dropout: {str(modality_dropout).lower()}
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


def test_plan_lines_forwards_extra_overrides_to_the_eval(tmp_path):
    train_folder = tmp_path / "mimicgen" / "run1"
    _write_train_cfg(train_folder)

    plan = plan_lines(str(train_folder), ["n_eval=100", "eval_batch_size=20"], reeval=False)

    _, overrides = plan[0]
    assert "n_eval=100" in overrides
    assert "eval_batch_size=20" in overrides


def test_plan_lines_does_not_duplicate_resolved_keys(tmp_path):
    train_folder = tmp_path / "mimicgen" / "run1"
    _write_train_cfg(train_folder)

    plan = plan_lines(str(train_folder), ["checkpoint=/custom/last.ckpt"], reeval=False)

    _, overrides = plan[0]
    assert [o for o in overrides if o.startswith("checkpoint=")] == ["checkpoint=/custom/last.ckpt"]


def _combo(overrides):
    return tuple(
        o.split("=", 1)[1]
        for key in ("rgb_static", "rgb_gripper", "language", "proprio")
        for o in overrides
        if o.startswith(f"eval_modalities.{key}=")
    )


def test_plan_lines_non_dropout_run_does_not_set_eval_modalities(tmp_path):
    train_folder = tmp_path / "mimicgen" / "run1"
    _write_train_cfg(train_folder)

    ((_, overrides),) = plan_lines(str(train_folder), [], reeval=False)

    assert not any(o.startswith("eval_modalities.") for o in overrides)


def test_plan_lines_dropout_run_sweeps_seven_token_combos_all_on_first(tmp_path):
    train_folder = tmp_path / "mimicgen" / "run1"
    _write_train_cfg(train_folder, modality_dropout=True)

    plan = plan_lines(str(train_folder), [], reeval=False)

    combos = [_combo(overrides) for _, overrides in plan]
    assert len(combos) == 7 == len(set(combos))
    assert combos[0] == ("True", "True", "True", "True")
    assert all(service == "eval-mimicgen" for service, _ in plan)
    assert all(c[3] == "True" for c in combos)  # no proprio to withhold


def test_plan_lines_dropout_run_with_proprio_sweeps_fourteen_combos(tmp_path):
    train_folder = tmp_path / "mimicgen" / "run1"
    _write_train_cfg(train_folder, modality_dropout=True, use_proprio=True)

    combos = [_combo(o) for _, o in plan_lines(str(train_folder), [], reeval=False)]

    assert len(combos) == 14 == len(set(combos))
    assert ("True", "True", "True", "False") in combos


def test_plan_lines_dropout_reeval_only_on_first_line(tmp_path):
    """Each eval-mimicgen line merges into the same result.csv; reeval on a later line
    would rotate away the combos the earlier lines just wrote."""
    train_folder = tmp_path / "mimicgen" / "run1"
    _write_train_cfg(train_folder, modality_dropout=True)

    plan = plan_lines(str(train_folder), [], reeval=True)

    assert ["reeval=True" in overrides for _, overrides in plan] == [True] + [False] * 6


def test_plan_lines_dropout_forwards_extra_overrides_and_sweep_wins(tmp_path):
    train_folder = tmp_path / "mimicgen" / "run1"
    _write_train_cfg(train_folder, modality_dropout=True)

    plan = plan_lines(str(train_folder), ["n_eval=100", "eval_modalities.language=False"], reeval=False)

    last_lang = []
    for _, overrides in plan:
        assert "n_eval=100" in overrides
        # Hydra applies the last occurrence of a key: the sweep's value must come last.
        last_lang.append([o for o in overrides if o.startswith("eval_modalities.language=")][-1])
    assert last_lang.count("eval_modalities.language=True") == 4
    assert last_lang.count("eval_modalities.language=False") == 3


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


def _fake_wandb(monkeypatch):
    logged = {}

    class _FakeArtifact:
        def __init__(self, name, type):
            self.files = []

        def add_file(self, path, name):
            self.files.append((path, name))

    class _FakeRun:
        def log_artifact(self, artifact):
            logged["files"] = artifact.files

        def finish(self):
            pass

    monkeypatch.setattr(mimicgen_pipeline.wandb, "init", lambda **kwargs: _FakeRun())
    monkeypatch.setattr(mimicgen_pipeline.wandb, "Artifact", _FakeArtifact)
    return logged


def _write_result_csv(train_folder: Path, combos) -> Path:
    checkpoint = str(train_folder / "seed_42" / "saved_models" / "last.ckpt")
    csv_path = result_dir(str(train_folder), checkpoint, "mimicgen", "core") / "result.csv"
    rows = [
        {
            "task_name": "square_d0",
            "success": 1,
            "use_rgb_static": static,
            "use_rgb_gripper": wrist,
            "use_language": lang,
            "use_proprio": 0,
        }
        for static, wrist, lang in combos
    ]
    write_csv(csv_path, rows)
    return csv_path


def test_upload_adds_pid_modality_txt_for_several_combos(tmp_path, monkeypatch):
    train_folder = tmp_path / "some_group" / "run1"
    _write_train_cfg(train_folder, modality_dropout=True)
    csv_path = _write_result_csv(train_folder, [(1, 1, 1), (1, 0, 1)])
    logged = _fake_wandb(monkeypatch)
    pid_calls = []

    def fake_run(cmd, **kwargs):
        pid_calls.append(cmd)
        return mimicgen_pipeline.subprocess.CompletedProcess(cmd, 0, stdout="PID REPORT\n", stderr="")

    monkeypatch.setattr(mimicgen_pipeline.subprocess, "run", fake_run)

    mimicgen_pipeline.upload(str(train_folder), [])

    pid_txt = csv_path.parent / "pid_modality.txt"
    assert pid_calls and pid_calls[0][-1] == str(csv_path)
    assert pid_txt.read_text() == "PID REPORT\n"
    assert logged["files"] == [(str(csv_path), "mimicgen.csv"), (str(pid_txt), "pid_modality.txt")]


def test_upload_skips_pid_for_a_single_combo(tmp_path, monkeypatch):
    train_folder = tmp_path / "some_group" / "run1"
    _write_train_cfg(train_folder)
    csv_path = _write_result_csv(train_folder, [(1, 1, 1)])
    logged = _fake_wandb(monkeypatch)
    monkeypatch.setattr(
        mimicgen_pipeline.subprocess, "run", lambda *a, **k: pytest.fail("pid_modality.py must not run")
    )

    mimicgen_pipeline.upload(str(train_folder), [])

    assert logged["files"] == [(str(csv_path), "mimicgen.csv")]


def test_upload_survives_pid_failure(tmp_path, monkeypatch, capsys):
    train_folder = tmp_path / "some_group" / "run1"
    _write_train_cfg(train_folder, modality_dropout=True)
    csv_path = _write_result_csv(train_folder, [(1, 1, 1), (0, 1, 1)])
    logged = _fake_wandb(monkeypatch)

    def failing_run(cmd, **kwargs):
        raise mimicgen_pipeline.subprocess.CalledProcessError(1, cmd, stderr="boom")

    monkeypatch.setattr(mimicgen_pipeline.subprocess, "run", failing_run)

    mimicgen_pipeline.upload(str(train_folder), [])

    assert logged["files"] == [(str(csv_path), "mimicgen.csv")]
    assert "pid_modality.py failed" in capsys.readouterr().err
