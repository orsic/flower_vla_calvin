#!/usr/bin/env bash
# Entry point for all podman operations in this repo.
# Usage:
#   ./run.sh build                        Build the container image (flower-vla-eval:latest)
#   ./run.sh shell                        Interactive bash session inside the container
#   ./run.sh download                     Download the LIBERO-10 eval checkpoint from HuggingFace
#   ./run.sh download-pret                Download the general pretrained checkpoint (for training)
#   ./run.sh download-data [bench|all]    Download LIBERO demo hdf5 files (default: libero_10)
#   ./run.sh train [bench] [...]          Fine-tune (default: libero_10, all modalities, ~15-22 h, 4 GPUs)
#                                          then auto-runs ./run.sh pipeline on the same GPUs (SKIP_PIPELINE=1 to skip)
#   ./run.sh train-frozen                 Ablation: frozen Florence VLM, action expert from scratch (no auto-pipeline)
#   ./run.sh train-dropout [bench] [...]  Fine-tune with modality-token dropout (default: libero_10); auto-pipeline too
#   ./run.sh train-dropout-resume [bench] [...] Resume train-dropout from CKPT_PATH; auto-pipeline too
#   ./run.sh eval                         Run the LIBERO-10 evaluation
#   ./run.sh download-plus                Download LIBERO-Plus simulation assets (run once before eval-plus)
#   ./run.sh eval-plus                    Run the LIBERO-Plus robustness evaluation (7 perturbation categories)
#   ./run.sh pipeline <train_run_dir> [...]  Full post-training eval (LIBERO[-Plus], modality sweep if dropout) + W&B upload
#   ./run.sh smoke                        Run the smoke test (verifies env before full eval)
#   ./run.sh devenv                       Regenerate .devcontainer/.env from vars.env (run after editing vars.env)
#
#   ./run.sh build-mimicgen               Build the MimicGen container image (flower-vla-mimicgen:latest)
#   ./run.sh download-mimicgen [ds|all]   Download MimicGen `core` demo hdf5 files (default: all)
#   ./run.sh prepare-mimicgen [ds|all] [-j N]  Render downloaded demos into image observations
#   ./run.sh rerender-mimicgen [-j N]     Download + render each dataset (re-renders stale camera sizes)
#   ./run.sh train-mimicgen [...]         Fine-tune on all MimicGen `core` datasets (CALVIN's training recipe)
#                                          then auto-runs ./run.sh pipeline-mimicgen (SKIP_PIPELINE=1 to skip)
#   ./run.sh eval-mimicgen                Run the MimicGen evaluation (all modalities)
#   ./run.sh pipeline-mimicgen <train_run_dir> [...]  Post-training MimicGen eval + W&B upload
#   ./run.sh analyze-mimicgen run|filter [...]  Success rate per task family x d0/d1/d2 from W&B
#
# Valid benchmarks: libero_10, libero_90, libero_spatial, libero_object, libero_goal
#
# Configure host-specific paths in vars.env (copied from vars.env.example).
# Shell environment variables take precedence over vars.env values.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT
COMPOSE="$REPO_ROOT/scripts/podman/compose.yml"
IMAGE="flower-vla-eval:latest"

# Load vars.env — shell env wins over file values.
if [[ -f "$REPO_ROOT/vars.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/vars.env"
    set +a
fi

# Replace the placeholder default so the volume mount doesn't fail.
if [[ "${LIBERO_HDF5_DIR:-/path/to/libero_hdf5}" == "/path/to/libero_hdf5" ]]; then
    LIBERO_HDF5_DIR="$REPO_ROOT/data/libero_hdf5"
fi
MIMICGEN_HDF5_DIR="${MIMICGEN_HDF5_DIR:-${DATA_DIR:-$REPO_ROOT/data}/mimicgen_hdf5}"

# Container hostname (compose.yml) so W&B/logs identify the host a run came from.
export HOST_HOSTNAME="${HOST_HOSTNAME:-${HOSTNAME:-$(uname -n)}}"
export LIBERO_HDF5_DIR
export MIMICGEN_HDF5_DIR

CMD="${1:-help}"
shift || true

# Short label for the modality combo a `train` invocation's Hydra overrides select, e.g.
# "model.modalities.language=False" -> "static+wrist" (what's left), or
# "model.use_proprio=True" -> "static+wrist+lang+proprio" (what's present). Empty string
# for the default combo (static+wrist+lang on, proprio off), so the default run-dir name
# is unchanged. Uses the same short names as scripts/compare_eval_csvs.py.
modality_label() {
    local static=1 wrist=1 lang=1 proprio=0
    for arg in "$@"; do
        case "$arg" in
            model.modalities.rgb_static=[Ff]*) static=0 ;;
            model.modalities.rgb_gripper=[Ff]*) wrist=0 ;;
            model.modalities.language=[Ff]*) lang=0 ;;
            model.use_proprio=[Tt]*) proprio=1 ;;
            model.modalities.proprio=[Ff]*) proprio=0 ;;
        esac
    done
    if [[ $static == 1 && $wrist == 1 && $lang == 1 && $proprio == 0 ]]; then
        echo ""
        return
    fi
    local parts=()
    [[ $static == 0 ]] || parts+=(static)
    [[ $wrist == 0 ]] || parts+=(wrist)
    [[ $lang == 0 ]] || parts+=(lang)
    [[ $proprio == 0 ]] || parts+=(proprio)
    (IFS=+; echo "${parts[*]}")
}

# Helper for train.
# Optionally takes a LIBERO benchmark as the first positional arg (no "=" = not a Hydra
# override); all remaining args are appended as Hydra overrides, including any
# model.modalities.* selecting a fixed modality-ablation combo (default: all three on).
# Valid benchmarks: libero_10 (default), libero_90, libero_spatial, libero_object, libero_goal
run_train() {
    local svc="$1"; shift
    local bench=libero_10
    if [[ $# -gt 0 && "$1" != *=* ]]; then bench="$1"; shift; fi
    local label
    label="$(modality_label "$@")"
    # Exported (not local): makes the training GPU set explicit in this process instead of
    # only implicit in compose.yml's train-service fallback, so a ./run.sh pipeline chained
    # after this (see the train/train-dropout* cases) reuses the same GPUs instead of
    # falling back to eval's own single-GPU default. No-op for the training container
    # itself -- same value it would have defaulted to anyway.
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
    # Not local: read back by the train/train-dropout* cases to chain ./run.sh pipeline.
    TRAIN_RUN_DIR="/saves/train_logs/${bench}${label:+_$label}/$(date +%Y-%m-%d_%H-%M-%S)"
    podman-compose -f "$COMPOSE" run --rm "$svc" \
        python flower/training_libero.py \
        "libero_benchmark=$bench" \
        "model.pretrained_model_path=/saves/checkpoints/flower_vla_pret/360000_model_weights.pt" \
        devices=-1 \
        log_dir=/saves/train_logs \
        num_workers=8 \
        seed=42 \
        "hydra.run.dir=$TRAIN_RUN_DIR" \
        "$@"
}

# Helper for train-dropout and train-dropout-resume.
# Optionally takes a LIBERO benchmark as the first positional arg (no "=" = not a Hydra
# override); all remaining args are appended as Hydra overrides.
# Valid benchmarks: libero_10 (default), libero_90, libero_spatial, libero_object, libero_goal
run_dropout_train() {
    local svc="$1"; shift
    local bench=libero_10
    if [[ $# -gt 0 && "$1" != *=* ]]; then bench="$1"; shift; fi
    # See run_train's comment: makes the training GPU set explicit for a chained pipeline.
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
    # Not local: read back by the train-dropout* cases to chain ./run.sh pipeline.
    TRAIN_RUN_DIR="/saves/train_logs/${bench}_dropout/$(date +%Y-%m-%d_%H-%M-%S)"
    podman-compose -f "$COMPOSE" run --rm "$svc" \
        python flower/training_libero.py \
        "libero_benchmark=$bench" \
        "model.pretrained_model_path=/saves/checkpoints/flower_vla_pret/360000_model_weights.pt" \
        model.modality_dropout=True \
        model.modality_dropout_keep_fraction=0.5 \
        "model.modality_dropout_alphas=[1.0,1.0,1.0]" \
        model.modality_dropout_proprio_keep_p=0.5 \
        devices=-1 \
        log_dir=/saves/train_logs \
        num_workers=8 \
        seed=42 \
        "hydra.run.dir=$TRAIN_RUN_DIR" \
        "$@"
}

# Helper for train-mimicgen. All remaining args are appended as Hydra overrides.
# Unlike run_train there is no benchmark positional arg -- a MimicGen run always trains
# on every dataset in flower.datasets.mimicgen_tasks.CORE_DATASETS.
run_train_mimicgen() {
    local svc="$1"; shift
    # See run_train's comment: makes the training GPU set explicit for a chained pipeline.
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
    # Not local: read back by the train-mimicgen case to chain ./run.sh pipeline-mimicgen.
    TRAIN_RUN_DIR="/saves/train_logs/mimicgen/$(date +%Y-%m-%d_%H-%M-%S)"
    podman-compose -f "$COMPOSE" run --rm "$svc" \
        python flower/training_libero.py \
        --config-name=config_mimicgen \
        "model.pretrained_model_path=/saves/checkpoints/flower_vla_pret/360000_model_weights.pt" \
        devices=-1 \
        log_dir=/saves/train_logs \
        num_workers=8 \
        seed=42 \
        "hydra.run.dir=$TRAIN_RUN_DIR" \
        "$@"
}

case "$CMD" in
  build)
    podman build \
        --tag "$IMAGE" \
        --file "$REPO_ROOT/scripts/podman/Containerfile" \
        "$REPO_ROOT"
    ;;

  shell)
    podman-compose -f "$COMPOSE" run --rm shell
    ;;

  download)
    podman-compose -f "$COMPOSE" run --rm download
    ;;

  download-pret)
    podman-compose -f "$COMPOSE" run --rm download-pret
    ;;

  download-data)
    # Optional first arg: benchmark name or "all" (default: libero_10).
    # "all" fetches every suite; a benchmark name fetches only that suite's hdf5 files.
    local_bench="${1:-libero_10}"
    if [[ "$local_bench" == "all" ]]; then
        INCLUDE_ARGS=()
    else
        INCLUDE_ARGS=(--include "${local_bench}/*")
    fi
    podman-compose -f "$COMPOSE" run --rm download-data \
        huggingface-cli download yifengzhu-hf/LIBERO-datasets \
        --repo-type dataset \
        "${INCLUDE_ARGS[@]}" \
        --local-dir /libero_hdf5
    ;;

  train)
    # Optional: first arg is the benchmark (default libero_10), remaining are Hydra overrides.
    # A fixed modality-ablation combo is selected with model.modalities.* overrides; default
    # is all three modalities on (identical to previous behavior). Examples:
    #   ./run.sh train libero_90
    #   ./run.sh train libero_10 model.modalities.language=False
    # On success, chains straight into ./run.sh pipeline on the same GPUs training used.
    # Set SKIP_PIPELINE=1 to queue several trainings without waiting on each one's eval.
    run_train train "$@"
    # PIPELINE_REEVAL[_SUITES] is meaningless for a run that has no prior eval results
    # yet, and would otherwise leak in from vars.env and abort a freshly-trained run's
    # chained pipeline if its suite names don't match this benchmark -- clear both.
    [[ -n "${SKIP_PIPELINE:-}" ]] || PIPELINE_REEVAL= PIPELINE_REEVAL_SUITES= "$0" pipeline "$TRAIN_RUN_DIR"
    ;;

  train-frozen)
    podman-compose -f "$COMPOSE" run --rm train-frozen-expert
    ;;

  train-dropout)
    # Optional: first arg is the benchmark (default libero_10), remaining are Hydra overrides.
    # Examples:
    #   ./run.sh train-dropout libero_90
    #   ./run.sh train-dropout libero_spatial model.modality_dropout_keep_fraction=0.7
    # On success, chains straight into ./run.sh pipeline on the same GPUs training used.
    # Set SKIP_PIPELINE=1 to queue several trainings without waiting on each one's eval.
    run_dropout_train train-dropout "$@"
    # PIPELINE_REEVAL[_SUITES] is meaningless for a run that has no prior eval results
    # yet, and would otherwise leak in from vars.env and abort a freshly-trained run's
    # chained pipeline if its suite names don't match this benchmark -- clear both.
    [[ -n "${SKIP_PIPELINE:-}" ]] || PIPELINE_REEVAL= PIPELINE_REEVAL_SUITES= "$0" pipeline "$TRAIN_RUN_DIR"
    ;;

  train-dropout-resume)
    # Resume from CKPT_PATH env var. Optional benchmark arg same as train-dropout.
    # Example: CKPT_PATH=/saves/.../last.ckpt ./run.sh train-dropout-resume libero_90
    # On success, chains straight into ./run.sh pipeline on the same GPUs training used.
    # Set SKIP_PIPELINE=1 to queue several trainings without waiting on each one's eval.
    run_dropout_train train-dropout-resume "$@"
    # PIPELINE_REEVAL[_SUITES] is meaningless for a run that has no prior eval results
    # yet, and would otherwise leak in from vars.env and abort a freshly-trained run's
    # chained pipeline if its suite names don't match this benchmark -- clear both.
    [[ -n "${SKIP_PIPELINE:-}" ]] || PIPELINE_REEVAL= PIPELINE_REEVAL_SUITES= "$0" pipeline "$TRAIN_RUN_DIR"
    ;;

  eval)
    # Build the full python command so user-provided Hydra overrides ("$@") are
    # appended and take precedence over defaults. podman-compose run SERVICE CMD
    # replaces the service's command entirely, so we must always pass the full cmd.
    podman-compose -f "$COMPOSE" run --rm eval \
        python flower/evaluation/flower_eval_libero.py \
        benchmark_name=libero_10 \
        train_folder=/saves/checkpoints/libero_10 \
        checkpoint=/saves/checkpoints/libero_10 \
        dataset_path=/workspace \
        log_dir=/saves/eval_logs \
        n_eval=20 \
        num_videos=3 \
        log_wandb=True \
        "hydra.run.dir=/saves/hydra_outputs/$(date +%Y-%m-%d_%H-%M-%S)" \
        "$@"
    ;;

  download-plus)
    podman-compose -f "$COMPOSE" run --rm download-plus
    ;;

  eval-plus)
    # LIBERO-Plus robustness eval. Pass Hydra overrides as extra args, e.g.:
    #   ./run.sh eval-plus task_category="Camera Viewpoints" checkpoint=/saves/.../best.ckpt
    # Runs with LIBERO_VARIANT=plus (set in the eval-plus compose service).
    podman-compose -f "$COMPOSE" run --rm eval-plus \
        python flower/evaluation/flower_eval_libero.py \
        --config-name=eval_libero_plus \
        "hydra.run.dir=/saves/hydra_outputs/$(date +%Y-%m-%d_%H-%M-%S)" \
        "$@"
    ;;

  pipeline)
    # Post-training evaluation for one completed training run: LIBERO (all modality
    # combos for a dropout run, full-modality otherwise) + LIBERO-Plus (full-modality,
    # plus one eval per withheld modality: rgb_gripper/rgb_static/proprio/language),
    # then upload the result.csv files + a scripts/pid_modality.py summary onto that
    # run's W&B artifact. Trailing Hydra overrides reach both the planner and every eval
    # it launches, e.g.:
    #   ./run.sh pipeline /saves/train_logs/libero_10_dropout/2026-09-08_10-00-00
    #   ./run.sh pipeline /saves/train_logs/libero_10/2026-09-08_10-00-00 eval_batch_size=32 n_eval=10
    #   PIPELINE_RESUME=1 ./run.sh pipeline /saves/train_logs/.../<run>   # skip evaluated combos
    #   PIPELINE_REEVAL=1 ./run.sh pipeline /saves/train_logs/.../<run>   # discard + redo everything
    #   PIPELINE_REEVAL=1 PIPELINE_REEVAL_SUITES=plus_libero_10 ./run.sh pipeline /saves/train_logs/.../<run>
    #                      # discard + redo only plus_libero_10; orig_libero_10 untouched
    #   PIPELINE_REEVAL=1 PIPELINE_REEVAL_SUITES=plus_libero_10_no_lang ./run.sh pipeline /saves/train_logs/.../<run>
    #                      # discard + redo only the language-withheld LIBERO-Plus eval
    #   PIPELINE_SKIP_MODALITY_OFF=1 ./run.sh pipeline /saves/train_logs/.../<run>
    #                      # skip the 4 modality-withheld LIBERO-Plus evals this run
    #                      # (they make the LIBERO-Plus portion ~5x longer); catch them
    #                      # up later with PIPELINE_REEVAL_SUITES=plus_<bench>_no_<modality>
    if [[ $# -lt 1 ]]; then
        echo "Usage: ./run.sh pipeline <train_run_dir> [hydra_overrides...]" >&2
        exit 1
    fi
    train_dir="$1"; shift
    resume_flag=()
    [[ -n "${PIPELINE_RESUME:-}" ]] && resume_flag=(--resume)
    reeval_flag=()
    if [[ -n "${PIPELINE_REEVAL:-}" ]]; then
        reeval_flag=(--reeval)
        if [[ -n "${PIPELINE_REEVAL_SUITES:-}" ]]; then
            reeval_flag+=(--reeval-suites "$PIPELINE_REEVAL_SUITES")
        fi
    fi
    skip_modality_off_flag=()
    [[ -n "${PIPELINE_SKIP_MODALITY_OFF:-}" ]] && skip_modality_off_flag=(--skip-modality-off)

    plan="$(podman-compose -f "$COMPOSE" run --rm -T shell \
        python scripts/eval_pipeline.py plan --train-folder "$train_dir" \
        "${resume_flag[@]}" "${reeval_flag[@]}" "${skip_modality_off_flag[@]}" -- "$@")"

    # Read every plan line into memory before launching anything -- a long-running eval
    # container previously held the loop's plan text on a shared fd for its entire
    # lifetime (hours, for a full sweep), and that fd ended up consumed/closed partway
    # through, so only the first line ever ran. A plain array has no descriptor left
    # open once this line finishes, so nothing a child process does afterward can
    # affect the remaining iterations.
    mapfile -t plan_lines <<< "$plan"

    i=0
    for line in "${plan_lines[@]}"; do
        IFS=$'\t' read -r -a fields <<< "$line"
        [[ ${#fields[@]} -eq 0 ]] && continue
        svc="${fields[0]}"
        overrides=("${fields[@]:1}")
        i=$((i + 1))
        config_arg=()
        [[ "$svc" == "eval-plus" ]] && config_arg=(--config-name=eval_libero_plus)
        podman-compose -f "$COMPOSE" run --rm -T "$svc" \
            python flower/evaluation/flower_eval_libero.py \
            "${config_arg[@]}" \
            "${overrides[@]}" \
            "hydra.run.dir=/saves/hydra_outputs/$(date +%Y-%m-%d_%H-%M-%S)_${i}"
    done

    podman-compose -f "$COMPOSE" run --rm -T pipeline-artifacts \
        python scripts/eval_pipeline.py upload --train-folder "$train_dir" -- "$@"
    ;;

  build-mimicgen)
    podman build \
        --tag flower-vla-mimicgen:latest \
        --file "$REPO_ROOT/scripts/podman/Containerfile.mimicgen" \
        "$REPO_ROOT"
    ;;

  download-mimicgen)
    # Optional first arg: dataset name (e.g. square_d0) or "all" (default).
    local_ds="${1:-all}"
    if [[ "$local_ds" == "all" ]]; then
        INCLUDE_ARGS=(--include "core/*")
    else
        INCLUDE_ARGS=(--include "core/${local_ds}.hdf5")
    fi
    podman-compose -f "$COMPOSE" run --rm download-mimicgen \
        huggingface-cli download amandlek/mimicgen_datasets \
        --repo-type dataset \
        "${INCLUDE_ARGS[@]}" \
        --local-dir /mimicgen_hdf5/source
    ;;

  prepare-mimicgen)
    # Renders downloaded demos into image observations. Args pass straight through to
    # scripts/prepare_mimicgen.py: [dataset|all] [-j N] [--n-demo N].
    podman-compose -f "$COMPOSE" run --rm prepare-mimicgen \
        python scripts/prepare_mimicgen.py "$@"
    ;;

  rerender-mimicgen)
    # Download + render every dataset, one dataset at a time per job (-j N jobs in
    # parallel): prepare-mimicgen deletes each source after rendering it, so peak extra
    # disk stays ~N datasets' worth rather than the full ~95 GB source release. Existing
    # outputs at the current camera sizes are skipped; older ones are re-rendered in place.
    jobs=1
    if [[ "${1:-}" == "-j" ]]; then jobs="$2"; fi
    datasets="$(podman-compose -f "$COMPOSE" run --rm -T shell-mimicgen \
        python -c "from flower.datasets.mimicgen_tasks import CORE_DATASETS; print(*CORE_DATASETS)" | tail -n 1)"
    printf '%s\n' $datasets | xargs -P "$jobs" -I{} \
        bash -c 'bash "$0" download-mimicgen "$1" && bash "$0" prepare-mimicgen "$1"' "$REPO_ROOT/run.sh" {}
    ;;

  train-mimicgen)
    # Fine-tune on every MimicGen `core` dataset with CALVIN's training recipe
    # (conf/config_mimicgen.yaml).
    # On success, chains straight into ./run.sh pipeline-mimicgen on the same GPUs.
    # Set SKIP_PIPELINE=1 to skip.
    run_train_mimicgen train-mimicgen "$@"
    [[ -n "${SKIP_PIPELINE:-}" ]] || PIPELINE_REEVAL= "$0" pipeline-mimicgen "$TRAIN_RUN_DIR"
    ;;

  eval-mimicgen)
    podman-compose -f "$COMPOSE" run --rm eval-mimicgen \
        python flower/evaluation/flower_eval_mimicgen.py \
        train_folder=/saves/checkpoints/mimicgen \
        checkpoint=/saves/checkpoints/mimicgen \
        log_dir=/saves/eval_logs \
        n_eval=20 \
        num_videos=0 \
        "hydra.run.dir=/saves/hydra_outputs/$(date +%Y-%m-%d_%H-%M-%S)" \
        "$@"
    ;;

  pipeline-mimicgen)
    # Post-training evaluation for one completed MimicGen training run: a single
    # all-modalities eval across every dataset in mimicgen_tasks.CORE_DATASETS, then
    # uploads result.csv to that run's W&B artifact. Trailing Hydra overrides reach
    # both the planner and the eval it launches.
    #   ./run.sh pipeline-mimicgen /saves/train_logs/mimicgen/2026-09-08_10-00-00
    #   PIPELINE_REEVAL=1 ./run.sh pipeline-mimicgen /saves/train_logs/mimicgen/.../<run>
    if [[ $# -lt 1 ]]; then
        echo "Usage: ./run.sh pipeline-mimicgen <train_run_dir> [hydra_overrides...]" >&2
        exit 1
    fi
    train_dir="$1"; shift
    reeval_flag=()
    [[ -n "${PIPELINE_REEVAL:-}" ]] && reeval_flag=(--reeval)

    plan="$(podman-compose -f "$COMPOSE" run --rm -T shell-mimicgen \
        python scripts/mimicgen_pipeline.py plan --train-folder "$train_dir" \
        "${reeval_flag[@]}" -- "$@")"

    # See ./run.sh pipeline's identical comment: read the whole plan into memory before
    # launching anything, so no long-lived eval container can affect later iterations.
    mapfile -t plan_lines <<< "$plan"

    i=0
    for line in "${plan_lines[@]}"; do
        IFS=$'\t' read -r -a fields <<< "$line"
        [[ ${#fields[@]} -eq 0 ]] && continue
        svc="${fields[0]}"
        overrides=("${fields[@]:1}")
        i=$((i + 1))
        podman-compose -f "$COMPOSE" run --rm -T "$svc" \
            python flower/evaluation/flower_eval_mimicgen.py \
            "${overrides[@]}" \
            "hydra.run.dir=/saves/hydra_outputs/$(date +%Y-%m-%d_%H-%M-%S)_${i}"
    done

    podman-compose -f "$COMPOSE" run --rm -T pipeline-artifacts-mimicgen \
        python scripts/mimicgen_pipeline.py upload --train-folder "$train_dir" -- "$@"
    ;;

  analyze-mimicgen)
    # Read the mimicgen.csv member scripts/mimicgen_pipeline.py's upload attached to one or
    # more training runs, and report success rate per task family x difficulty variant.
    #   ./run.sh analyze-mimicgen run <run_id> [<run_id>...]
    #   ./run.sh analyze-mimicgen filter --filters '{"config.use_proprio": true}'
    # Needs WANDB_API_KEY set (vars.env or shell env) since this only reads from W&B.
    podman-compose -f "$COMPOSE" run --rm -T pipeline-artifacts-mimicgen \
        python scripts/mimicgen_sr.py "$@"
    ;;

  analyze)
    # Read the "evaluation" W&B artifact scripts/eval_pipeline.py's upload attached to
    # one or more training runs, and report PID + per-perturbation success rates.
    #   ./run.sh analyze run <run_id> [<run_id>...]
    #   ./run.sh analyze filter --filters '{"config.modality_dropout": true}'
    # Needs WANDB_API_KEY set (vars.env or shell env) since this only reads from W&B.
    podman-compose -f "$COMPOSE" run --rm -T pipeline-artifacts \
        python scripts/analyze_wandb.py "$@"
    ;;

  plot)
    # Paper figures from the same W&B evaluation artifacts analyze reads.
    #   ./run.sh plot presence --filters '{"config.modality_dropout": true}'
    #   ./run.sh plot perturbation --filters '...'
    #   ./run.sh plot severity --filters '...'
    #   ./run.sh plot all --filters '...'
    # Needs WANDB_API_KEY set (vars.env or shell env) since this only reads from W&B.
    podman-compose -f "$COMPOSE" run --rm -T pipeline-artifacts \
        python scripts/plot_eval.py "$@"
    ;;

  smoke)
    podman-compose -f "$COMPOSE" run --rm shell python scripts/smoke_test.py
    ;;

  devenv)
    sed \
        -e "s|@@HF_HOME@@|$HF_HOME|g" \
        -e "s|@@SAVES_DIR@@|$SAVES_DIR|g" \
        -e "s|@@LIBERO_HDF5_DIR@@|$LIBERO_HDF5_DIR|g" \
        "$REPO_ROOT/.devcontainer/devcontainer.json.template" \
        > "$REPO_ROOT/.devcontainer/devcontainer.json"
    echo "Written .devcontainer/devcontainer.json"
    ;;

  help|*)
    echo "Usage: ./run.sh <build|shell|download|download-pret|download-data|download-plus|train|train-frozen|train-dropout|eval|eval-plus|pipeline|analyze|plot|smoke|devenv>"
    echo ""
    echo "  build              Build the container image (flower-vla-eval:latest)"
    echo "  shell              Interactive bash inside the container"
    echo "  download           Download LIBERO-10 eval checkpoint from HuggingFace"
    echo "  download-pret      Download general pretrained checkpoint (for fine-tuning)"
    echo "  download-data [bench|all]    Download LIBERO demo hdf5 files (default: libero_10)"
    echo "                               bench: libero_10 | libero_90 | libero_spatial | libero_object | libero_goal"
    echo "                               all: download every suite"
    echo "  download-plus      Download LIBERO-Plus simulation assets (3D objects/textures)"
    echo "  train [bench] [hydra_overrides...]"
    echo "                     Fine-tune, 4 GPUs, ~15-22 h (default: libero_10, all modalities)"
    echo "                     bench: libero_10 | libero_90 | libero_spatial | libero_object | libero_goal"
    echo "                     Fixed modality ablation: model.modalities.rgb_static|rgb_gripper|language=False"
    echo "                     On success, auto-runs ./run.sh pipeline on the same GPUs training used."
    echo "                     Set SKIP_PIPELINE=1 to queue several trainings without waiting on each eval."
    echo "                     Example: ./run.sh train libero_90"
    echo "                              ./run.sh train libero_10 model.modalities.language=False"
    echo "  train-frozen       Ablation: frozen Florence VLM, action expert trained from random init"
    echo "                     (no auto-pipeline; run ./run.sh pipeline <dir> manually afterward)"
    echo "  train-dropout [bench] [hydra_overrides...]"
    echo "                     Fine-tune with Dirichlet modality-token dropout (default: libero_10)"
    echo "                     bench: libero_10 | libero_90 | libero_spatial | libero_object | libero_goal"
    echo "                     On success, auto-runs ./run.sh pipeline on the same GPUs training used"
    echo "                     (SKIP_PIPELINE=1 to skip)."
    echo "                     Example: ./run.sh train-dropout libero_90"
    echo "                              ./run.sh train-dropout libero_spatial model.modality_dropout_keep_fraction=0.7"
    echo "  train-dropout-resume [bench] [hydra_overrides...]"
    echo "                     Resume train-dropout from CKPT_PATH env var; auto-runs the pipeline too."
    echo "                     Example: CKPT_PATH=/saves/.../last.ckpt ./run.sh train-dropout-resume libero_90"
    echo "  eval               Run LIBERO-10 evaluation (./run.sh download first)"
    echo "  eval-plus          Run LIBERO-Plus robustness eval (./run.sh download-plus first)"
    echo "                     Pass Hydra overrides: task_category=\"Camera Viewpoints\" checkpoint=/saves/..."
    echo "  pipeline <train_run_dir> [hydra_overrides...]"
    echo "                     Full post-training eval for one run: LIBERO + LIBERO-Plus, or (if the"
    echo "                     run used model.modality_dropout) every modality combo on LIBERO + full"
    echo "                     LIBERO-Plus -- plus one more LIBERO-Plus eval per withheld modality"
    echo "                     (rgb_gripper/rgb_static/proprio/language), each its own result.csv."
    echo "                     Uploads result.csv + scripts/pid_modality.py output to the training"
    echo "                     run's W&B artifact. Trailing overrides reach every eval launched."
    echo "                     Example: ./run.sh pipeline /saves/train_logs/libero_10_dropout/2026-.../"
    echo "                              PIPELINE_RESUME=1 ./run.sh pipeline /saves/train_logs/.../<run>"
    echo "                     PIPELINE_REEVAL=1 [PIPELINE_REEVAL_SUITES=orig_<bench>,plus_<bench>,"
    echo "                     plus_<bench>_no_<modality>] backs up (results_<mtime>.csv) and re-runs"
    echo "                     the named suites from empty instead of merging into them; omitting the"
    echo "                     suites list means all of them."
    echo "                     PIPELINE_SKIP_MODALITY_OFF=1 skips the 4 modality-withheld LIBERO-Plus"
    echo "                     evals (they make the LIBERO-Plus portion ~5x longer); catch them up"
    echo "                     later with PIPELINE_REEVAL_SUITES=plus_<bench>_no_<modality>."
    echo "  analyze <run|filter> ..."
    echo "                     Read the W&B evaluation artifact(s) ./run.sh pipeline uploaded and report"
    echo "                     PID + per-perturbation success rates. Needs WANDB_API_KEY."
    echo "                     analyze run <run_id> [<run_id>...]     -- one report per run"
    echo "                     analyze filter --filters '<mongo-json>' -- mean/min/max across matches"
    echo "                     Example: ./run.sh analyze run libero_10_dropout_2026-09-09_13-56-20"
    echo "                              ./run.sh analyze filter --filters '{\"config.modality_dropout\": true}'"
    echo "  plot <presence|perturbation|severity|all> ..."
    echo "                     Paper figures (PDF) from the same W&B evaluation artifacts, sharing one"
    echo "                     --filters run pool across every figure. Needs WANDB_API_KEY."
    echo "                     Example: ./run.sh plot all --filters '{\"config.modality_dropout\": true}'"
    echo "  smoke              Quick sanity check: CUDA + imports + model load + 1 env step"
    echo "  devenv             Regenerate .devcontainer/.env from vars.env"
    echo ""
    echo "  build-mimicgen     Build the MimicGen container image (flower-vla-mimicgen:latest)"
    echo "  download-mimicgen [dataset|all]"
    echo "                     Download MimicGen \`core\` demo hdf5 files (default: all 26 datasets)"
    echo "                     Example: ./run.sh download-mimicgen square_d0"
    echo "  prepare-mimicgen [dataset|all] [-j N] [--n-demo N]"
    echo "                     Render downloaded demos into image observations (default: all, 100 demos)"
    echo "                     -j N renders N datasets concurrently. KEEP_MIMICGEN_SOURCE=1 keeps sources."
    echo "  rerender-mimicgen [-j N]"
    echo "                     download-mimicgen + prepare-mimicgen per dataset, N datasets at a time;"
    echo "                     re-renders outputs whose camera sizes are out of date"
    echo "  train-mimicgen [hydra_overrides...]"
    echo "                     Fine-tune on every MimicGen \`core\` dataset with CALVIN's training recipe."
    echo "                     On success, auto-runs ./run.sh pipeline-mimicgen on the same GPUs"
    echo "                     (SKIP_PIPELINE=1 to skip)."
    echo "  eval-mimicgen      Run the MimicGen evaluation, all modalities on"
    echo "  pipeline-mimicgen <train_run_dir> [hydra_overrides...]"
    echo "                     Post-training MimicGen eval (all modalities, every dataset) + W&B upload."
    echo "                     Example: ./run.sh pipeline-mimicgen /saves/train_logs/mimicgen/2026-.../"
    echo "                              PIPELINE_REEVAL=1 ./run.sh pipeline-mimicgen /saves/.../<run>"
    if [[ "$CMD" != "help" ]]; then
        exit 1
    fi
    ;;
esac
