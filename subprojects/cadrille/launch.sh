#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
source "${repo_root}/external/pins.env"

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <mesh-path> <out-dir> [extra python args...]" >&2
  exit 1
fi

mesh_path="$1"
out_dir="$2"
shift 2

image="meshxcad/cadrille:2026-03-13"
if ! docker image inspect "${image}" >/dev/null 2>&1; then
  docker build -t "${image}" -f "${script_dir}/Dockerfile" "${repo_root}"
fi
mkdir -p "${repo_root}/.cache/huggingface"

exec docker run --rm --gpus all \
  -e HF_HOME=/workspace/.cache/huggingface \
  -e CADRILLE_HOME=/opt/cadrille \
  -v "${repo_root}:/workspace" \
  "${image}" \
  python3 /workspace/subprojects/cadrille/run_cadrille.py \
    --mesh "${mesh_path}" \
    --out-dir "${out_dir}" \
    --checkpoint-path "${CADRILLE_MODEL}" \
    "$@"
