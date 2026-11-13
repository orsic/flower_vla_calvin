"""scripts/proprio_mean.py decides whether a run needs its mean proprio vector computed:
never for a model without proprio, and not again once the file exists."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from proprio_mean import target  # noqa: E402


def _run(tmp_path, use_proprio):
    run = tmp_path / "calvin_abcd_d" / "run"
    (run / ".hydra").mkdir(parents=True)
    (run / ".hydra" / "config.yaml").write_text(f"seed: 42\nmodel:\n  use_proprio: {str(use_proprio).lower()}\n")
    return run


def test_target_skips_a_model_without_proprio(tmp_path):
    assert target(str(_run(tmp_path, False)), None) is None


def test_target_is_the_eval_logs_file_until_it_exists(tmp_path):
    run = _run(tmp_path, True)
    path = target(str(run), None)
    assert path == run / "eval_logs" / "last" / "proprio_mean.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"pt")
    assert target(str(run), None) is None
