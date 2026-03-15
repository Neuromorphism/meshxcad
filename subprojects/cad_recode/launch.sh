#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
source "${repo_root}/external/pins.env"
mkdir -p "${repo_root}/.cache/huggingface"
export HF_HOME="${repo_root}/.cache/huggingface"

engine="${CAD_RECODE_ENGINE:-venv}"
if [[ $# -lt 2 ]]; then
  echo "usage: $0 <mesh-path> <out-dir> [extra python args...]" >&2
  exit 1
fi

mesh_path="$1"
out_dir="$2"
shift 2

if [[ "${engine}" == "venv" ]]; then
  if [[ ! -x "${repo_root}/.venvs/cad-recode/bin/python" ]]; then
    "${script_dir}/bootstrap_venv.sh"
  fi
  exec "${repo_root}/.venvs/cad-recode/bin/python" \
    "${script_dir}/run_cad_recode.py" \
    --mesh "${mesh_path}" \
    --out-dir "${out_dir}" \
    --model-id "${CAD_RECODE_MODEL}" \
    "$@"
fi

if [[ "${engine}" == "docker" ]]; then
  image="meshxcad/cad-recode:2026-03-13"
  if ! docker image inspect "${image}" >/dev/null 2>&1; then
    docker build -t "${image}" -f "${script_dir}/Dockerfile" "${repo_root}"
  fi
  mkdir -p "${repo_root}/.cache/huggingface"
  exec docker run --rm --gpus all \
    -e HF_HOME=/workspace/.cache/huggingface \
    -v "${repo_root}:/workspace" \
    "${image}" \
    python3 /workspace/subprojects/cad_recode/run_cad_recode.py \
      --mesh "${mesh_path}" \
      --out-dir "${out_dir}" \
      --model-id "${CAD_RECODE_MODEL}" \
      "$@"
fi

echo "Unsupported CAD_RECODE_ENGINE=${engine}" >&2
exit 1
