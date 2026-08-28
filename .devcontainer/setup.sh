#!/usr/bin/env bash
# Devcontainer post-start setup.
# Replicates the LIBERO path config written by scripts/podman/entrypoint.sh.
# VS Code replaces the image ENTRYPOINT with its own server bootstrap, so
# entrypoint.sh doesn't run automatically — this script fills that gap.
set -euo pipefail

LIBERO_PKG=/workspace/LIBERO/libero/libero
mkdir -p /root/.libero
cat > /root/.libero/config.yaml << EOF
assets: ${LIBERO_PKG}/assets
bddl_files: ${LIBERO_PKG}/bddl_files
benchmark_root: ${LIBERO_PKG}
datasets: /libero_hdf5
init_states: ${LIBERO_PKG}/init_files
EOF

echo "[devcontainer] LIBERO paths configured at /root/.libero/config.yaml"
