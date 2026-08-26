#!/usr/bin/env bash
# Container entrypoint: configures LIBERO paths and Python path, then runs CMD.
set -euo pipefail

# LIBERO reads its data paths from ~/.libero/config.yaml at import time.
# Without this file it runs an interactive prompt, which breaks non-interactive containers.
LIBERO_PKG=/workspace/LIBERO/libero/libero
mkdir -p /root/.libero
cat > /root/.libero/config.yaml << EOF
assets: ${LIBERO_PKG}/assets
bddl_files: ${LIBERO_PKG}/bddl_files
benchmark_root: ${LIBERO_PKG}
datasets: /libero_hdf5
init_states: ${LIBERO_PKG}/init_files
EOF

# LIBERO/libero/ is a namespace package (no __init__.py), so pip's editable-install
# finder doesn't resolve it. Setting PYTHONPATH directly makes both packages importable
# without any pip install step.
export PYTHONPATH=/workspace/LIBERO:/workspace

exec "$@"
