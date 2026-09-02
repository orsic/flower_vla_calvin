#!/usr/bin/env bash
# Entry point for all podman operations in this repo.
# Usage:
#   ./run.sh build                        Build the container image (flower-vla-eval:latest)
#   ./run.sh shell                        Interactive bash session inside the container
#   ./run.sh download                     Download the LIBERO-10 eval checkpoint from HuggingFace
#   ./run.sh download-pret                Download the general pretrained checkpoint (for training)
#   ./run.sh download-data [bench|all]    Download LIBERO demo hdf5 files (default: libero_10)
#   ./run.sh train [bench] [...]          Fine-tune (default: libero_10, all modalities, ~15-22 h, 4 GPUs)
#   ./run.sh train-frozen                 Ablation: frozen Florence VLM, action expert from scratch
#   ./run.sh train-dropout [bench] [...]  Fine-tune with modality-token dropout (default: libero_10)
#   ./run.sh train-dropout-resume [bench] [...] Resume train-dropout from CKPT_PATH
#   ./run.sh eval                         Run the LIBERO-10 evaluation
#   ./run.sh download-plus                Download LIBERO-Plus simulation assets (run once before eval-plus)
#   ./run.sh eval-plus                    Run the LIBERO-Plus robustness evaluation (7 perturbation categories)
#   ./run.sh smoke                        Run the smoke test (verifies env before full eval)
#   ./run.sh devenv                       Regenerate .devcontainer/.env from vars.env (run after editing vars.env)
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
export LIBERO_HDF5_DIR

CMD="${1:-help}"
shift || true

# Short label for the modality combo a `train` invocation's Hydra overrides select, e.g.
# "model.modalities.language=False" -> "static+wrist". Empty string when all three (or none
# of the model.modalities.* keys) are overridden, so the default run-dir name is unchanged.
# Uses the same short names and "+".join(sorted(...)) convention as scripts/compare_eval_csvs.py.
modality_label() {
    local static=1 wrist=1 lang=1
    for arg in "$@"; do
        case "$arg" in
            model.modalities.rgb_static=[Ff]*) static=0 ;;
            model.modalities.rgb_gripper=[Ff]*) wrist=0 ;;
            model.modalities.language=[Ff]*) lang=0 ;;
        esac
    done
    local parts=()
    [[ $static == 1 ]] || parts+=(static)
    [[ $wrist == 1 ]] || parts+=(wrist)
    [[ $lang == 1 ]] || parts+=(lang)
    if [[ ${#parts[@]} -eq 0 || ${#parts[@]} -eq 3 ]]; then
        echo ""
    else
        (IFS=+; echo "${parts[*]}")
    fi
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
    podman-compose -f "$COMPOSE" run --rm "$svc" \
        python flower/training_libero.py \
        "libero_benchmark=$bench" \
        "model.pretrained_model_path=/saves/checkpoints/flower_vla_pret/360000_model_weights.pt" \
        devices=-1 \
        log_dir=/saves/train_logs \
        num_workers=8 \
        seed=42 \
        "hydra.run.dir=/saves/train_logs/${bench}${label:+_$label}/$(date +%Y-%m-%d_%H-%M-%S)" \
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
    podman-compose -f "$COMPOSE" run --rm "$svc" \
        python flower/training_libero.py \
        "libero_benchmark=$bench" \
        "model.pretrained_model_path=/saves/checkpoints/flower_vla_pret/360000_model_weights.pt" \
        model.modality_dropout=True \
        model.modality_dropout_keep_fraction=0.5 \
        "model.modality_dropout_alphas=[1.0,1.0,1.0]" \
        rollout_lh_skip_epochs=39 \
        devices=-1 \
        log_dir=/saves/train_logs \
        num_workers=8 \
        seed=42 \
        "hydra.run.dir=/saves/train_logs/${bench}_dropout/$(date +%Y-%m-%d_%H-%M-%S)" \
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
    run_train train "$@"
    ;;

  train-frozen)
    podman-compose -f "$COMPOSE" run --rm train-frozen-expert
    ;;

  train-dropout)
    # Optional: first arg is the benchmark (default libero_10), remaining are Hydra overrides.
    # Examples:
    #   ./run.sh train-dropout libero_90
    #   ./run.sh train-dropout libero_spatial model.modality_dropout_keep_fraction=0.7
    run_dropout_train train-dropout "$@"
    ;;

  train-dropout-resume)
    # Resume from CKPT_PATH env var. Optional benchmark arg same as train-dropout.
    # Example: CKPT_PATH=/saves/.../last.ckpt ./run.sh train-dropout-resume libero_90
    run_dropout_train train-dropout-resume "$@"
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
    echo "Usage: ./run.sh <build|shell|download|download-pret|download-data|download-plus|train|train-frozen|train-dropout|eval|eval-plus|smoke|devenv>"
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
    echo "                     Example: ./run.sh train libero_90"
    echo "                              ./run.sh train libero_10 model.modalities.language=False"
    echo "  train-frozen       Ablation: frozen Florence VLM, action expert trained from random init"
    echo "  train-dropout [bench] [hydra_overrides...]"
    echo "                     Fine-tune with Dirichlet modality-token dropout (default: libero_10)"
    echo "                     bench: libero_10 | libero_90 | libero_spatial | libero_object | libero_goal"
    echo "                     Example: ./run.sh train-dropout libero_90"
    echo "                              ./run.sh train-dropout libero_spatial model.modality_dropout_keep_fraction=0.7"
    echo "  train-dropout-resume [bench] [hydra_overrides...]"
    echo "                     Resume train-dropout from CKPT_PATH env var"
    echo "                     Example: CKPT_PATH=/saves/.../last.ckpt ./run.sh train-dropout-resume libero_90"
    echo "  eval               Run LIBERO-10 evaluation (./run.sh download first)"
    echo "  eval-plus          Run LIBERO-Plus robustness eval (./run.sh download-plus first)"
    echo "                     Pass Hydra overrides: task_category=\"Camera Viewpoints\" checkpoint=/saves/..."
    echo "  smoke              Quick sanity check: CUDA + imports + model load + 1 env step"
    echo "  devenv             Regenerate .devcontainer/.env from vars.env"
    if [[ "$CMD" != "help" ]]; then
        exit 1
    fi
    ;;
esac
