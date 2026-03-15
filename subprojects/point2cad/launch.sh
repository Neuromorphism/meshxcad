#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <mesh-path> <out-dir> [extra python args...]" >&2
  exit 1
fi

mesh_path="$1"
out_dir="$2"
shift 2

if [[ ! -x "${repo_root}/.venvs/point2cad/bin/python" ]]; then
  "${script_dir}/bootstrap_venv.sh"
fi

exec "${repo_root}/.venvs/point2cad/bin/python" \
  "${script_dir}/run_point2cad.py" \
  --mesh "${mesh_path}" \
  --out-dir "${out_dir}" \
  --engine docker \
  "$@"
