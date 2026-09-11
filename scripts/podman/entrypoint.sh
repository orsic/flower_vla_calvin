#!/usr/bin/env bash
# Container entrypoint: configures LIBERO paths and Python path, then runs CMD.
set -euo pipefail

# LIBERO reads its data paths from ~/.libero/config.yaml at import time.
# Without this file it runs an interactive prompt, which breaks non-interactive containers.
#
# LIBERO_VARIANT controls which LIBERO package is active:
#   orig (default) — upstream LIBERO submodule; original suites (libero_10 = 10 tasks)
#   plus           — LIBERO-Plus fork; expanded suites with perturbation variants
#                    (libero_10 = 2519 tasks across 7 perturbation categories)
LIBERO_VARIANT="${LIBERO_VARIANT:-orig}"
if [ "$LIBERO_VARIANT" = "plus" ]; then
    LIBERO_PKG=/workspace/LIBERO-plus/libero/libero
    export PYTHONPATH=/workspace/LIBERO-plus:/workspace
else
    LIBERO_PKG=/workspace/LIBERO/libero/libero
    export PYTHONPATH=/workspace/LIBERO:/workspace
fi

mkdir -p /root/.libero
cat > /root/.libero/config.yaml << EOF
assets: ${LIBERO_PKG}/assets
bddl_files: ${LIBERO_PKG}/bddl_files
benchmark_root: ${LIBERO_PKG}
datasets: /libero_hdf5
init_states: ${LIBERO_PKG}/init_files
EOF

exec "$@"
