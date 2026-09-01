#!/usr/bin/env bash
# Entry point for all podman operations in this repo.
# Usage:
#   ./run.sh build              Build the container image (flower-vla-eval:latest)
#   ./run.sh shell              Interactive bash session inside the container
#   ./run.sh download           Download the LIBERO-10 eval checkpoint from HuggingFace
#   ./run.sh download-pret      Download the general pretrained checkpoint (for training)
#   ./run.sh download-data      Download the LIBERO-10 demo hdf5 files (for training)
#   ./run.sh train              Fine-tune on LIBERO-10 (~15-22 h, 4 GPUs)
#   ./run.sh train-frozen       Ablation: frozen Florence VLM, action expert from scratch
#   ./run.sh eval               Run the LIBERO-10 evaluation
#   ./run.sh download-plus      Download LIBERO-Plus simulation assets (run once before eval-plus)
#   ./run.sh eval-plus          Run the LIBERO-Plus robustness evaluation (7 perturbation categories)
#   ./run.sh smoke              Run the smoke test (verifies env before full eval)
#   ./run.sh devenv             Regenerate .devcontainer/.env from vars.env (run after editing vars.env)
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
    podman-compose -f "$COMPOSE" run --rm download-data
    ;;

  train)
    podman-compose -f "$COMPOSE" run --rm train
    ;;

  train-frozen)
    podman-compose -f "$COMPOSE" run --rm train-frozen-expert
    ;;

  train-dropout)
    podman-compose -f "$COMPOSE" run --rm train-dropout
    ;;

  train-dropout-resume)
    podman-compose -f "$COMPOSE" run --rm train-dropout-resume
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
    echo "  download-data      Download LIBERO-10 demo hdf5 files (for fine-tuning)"
    echo "  download-plus      Download LIBERO-Plus simulation assets (3D objects/textures)"
    echo "  train              Fine-tune on LIBERO-10, 4 GPUs, ~15-22 h"
    echo "  train-frozen       Ablation: frozen Florence VLM, action expert trained from random init"
    echo "  train-dropout          Full fine-tune with Dirichlet modality-token dropout (3 groups)"
    echo "  train-dropout-resume   Resume train-dropout from CKPT_PATH (defaults to epoch-9 ckpt)"
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
