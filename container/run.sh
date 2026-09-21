#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${IMAGE:-complementarity-artifact:local}"

if (( $# == 0 )); then
    set -- ./scripts/reproduce.sh
fi

docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e XDG_CACHE_HOME=/tmp/.cache \
    -e MPLCONFIGDIR=/tmp/matplotlib \
    -v "${REPO_ROOT}:/workspace" \
    -w /workspace \
    "${IMAGE}" \
    "$@"
