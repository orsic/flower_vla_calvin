"""run.sh's VLA-Adapter cases drive the existing LIBERO/CALVIN harnesses with policy=vla_adapter
in the eval-vla-adapter service (stub podman-compose, as in tests/test_run_sh_calvin.py)."""

from tests.test_run_sh_calvin import _run


def test_eval_vla_adapter_runs_the_libero_harness_with_the_reference_resolution(tmp_path):
    lines = _run(tmp_path, ["eval-vla-adapter", "n_eval=2"])

    assert len(lines) == 1
    cmd = lines[0]
    assert " run --rm -T eval-vla-adapter python flower/evaluation/flower_eval_libero.py " in cmd
    for override in (
        "policy=vla_adapter",
        "benchmark_name=libero_10",
        "checkpoint=/saves/checkpoints/vla_adapter/LIBERO-Long-Pro",
        "env_img_res=256",
        "n_eval=50",
    ):
        assert f" {override} " in cmd
    assert cmd.endswith("n_eval=2")  # user overrides come last and win


def test_eval_calvin_vla_adapter_runs_the_calvin_harness(tmp_path):
    lines = _run(tmp_path, ["eval-calvin-vla-adapter", "num_sequences=5"])

    assert len(lines) == 1
    cmd = lines[0]
    assert " run --rm -T eval-vla-adapter python flower/evaluation/flower_evaluate.py " in cmd
    assert " policy=vla_adapter " in cmd
    assert " checkpoint=/saves/checkpoints/vla_adapter/CALVIN-ABC-Pro " in cmd
    assert cmd.endswith("num_sequences=5")


def test_download_vla_adapter_fetches_both_checkpoints(tmp_path):
    log = "\n".join(_run(tmp_path, ["download-vla-adapter"]))  # the -c script spans lines

    assert " run --rm -T shell-vla-adapter python -c " in log
    assert "LIBERO-Long-Pro" in log and "CALVIN-ABC-Pro" in log


def test_eval_vla_adapter_keeps_the_harness_warm_up(tmp_path):
    (cmd,) = _run(tmp_path, ["eval-vla-adapter"])
    assert "num_steps_wait" not in cmd and "wait_gripper" not in cmd


def test_pipeline_vla_adapter_runs_plan_every_line_then_collect(tmp_path):
    lines = _run(tmp_path, ["pipeline-vla-adapter", "calvin"], {"PIPELINE_RESUME": "1"})

    assert (
        " run --rm -T shell-vla-adapter python scripts/vla_adapter_pipeline.py plan --benchmark calvin --resume"
        in lines[0]
    )
    evals = lines[1:-1]
    assert len(evals) == 2
    assert all(
        " run --rm -T eval-vla-adapter python flower/evaluation/flower_evaluate.py policy=vla_adapter " in e
        for e in evals
    )
    assert "eval_modalities.rgb_static=False" in evals[1]
    assert (
        " run --rm -T shell-vla-adapter python scripts/vla_adapter_pipeline.py collect --benchmark calvin" in lines[-1]
    )


def test_pipeline_vla_adapter_passes_checkpoint_and_no_masked_to_plan_and_collect(tmp_path):
    lines = _run(tmp_path, ["pipeline-vla-adapter", "libero", "--checkpoint", "/saves/x--5_chkpt", "--no-masked"])
    assert "plan --benchmark libero --checkpoint /saves/x--5_chkpt --no-masked" in lines[0]
    assert "collect --benchmark libero --checkpoint /saves/x--5_chkpt --no-masked" in lines[-1]


def test_train_vla_adapter_dropout_runs_torchrun_per_gpu_then_evaluates_the_last_checkpoint(tmp_path):
    lines = _run(tmp_path, ["train-vla-adapter-dropout", "--max_steps", "50"], {"CUDA_VISIBLE_DEVICES": "0,1"})
    train = lines[0]
    assert " run --rm -T train-vla-adapter torchrun --standalone --nnodes 1 --nproc-per-node 2 " in train
    assert "flower/vla_adapter/finetune_dropout.py --modality_dropout subset " in train
    for flag in (
        "--batch_size 8",
        "--grad_accumulation_steps 1",
        "--lora_rank 64",
        "--learning_rate 2e-4",
        "--max_steps 200005",
        "--dataset_name libero_10_no_noops",
        "--use_pro_version True",
    ):
        assert f" {flag} " in train
    assert train.endswith("--max_steps 50")
    run_dir = train.split("--run_root_dir ")[1].split()[0]
    assert run_dir.startswith("/saves/train_logs/vla_adapter_libero_10_dropout/")
    plan = next(line for line in lines if "vla_adapter_pipeline.py plan" in line)
    assert f"--checkpoint {run_dir}/run--50_chkpt --no-masked" in plan


def test_train_vla_adapter_dropout_skip_pipeline(tmp_path):
    lines = _run(tmp_path, ["train-vla-adapter-dropout"], {"SKIP_PIPELINE": "1"})
    assert len(lines) == 1
