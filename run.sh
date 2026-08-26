#!/usr/bin/env bash
# Entry point for all podman operations in this repo.
# Usage:
#   ./run.sh build              Build the container image (flower-vla-eval:latest)
#   ./run.sh shell              Interactive bash session inside the container
#   ./run.sh download           Download the LIBERO-10 checkpoint from HuggingFace
#   ./run.sh eval               Run the LIBERO-10 evaluation
#   ./run.sh smoke              Run the smoke test (verifies env before full eval)
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

  eval)
    podman-compose -f "$COMPOSE" run --rm eval
    ;;

  smoke)
    podman-compose -f "$COMPOSE" run --rm shell python scripts/smoke_test.py
    ;;

  help|*)
    echo "Usage: ./run.sh <build|shell|download|eval|smoke>"
    echo ""
    echo "  build     Build the container image (flower-vla-eval:latest)"
    echo "  shell     Interactive bash inside the container"
    echo "  download  Download LIBERO-10 checkpoint from HuggingFace"
    echo "  eval      Run LIBERO-10 evaluation (./run.sh download first)"
    echo "  smoke     Quick sanity check: CUDA + imports + model load + 1 env step"
    if [[ "$CMD" != "help" ]]; then
        exit 1
    fi
    ;;
esac
