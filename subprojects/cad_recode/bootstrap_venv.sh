#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
venv_dir="${repo_root}/.venvs/cad-recode"

if command -v uv >/dev/null 2>&1; then
  uv venv --clear "${venv_dir}"
  uv pip install --python "${venv_dir}/bin/python" --upgrade pip wheel
  uv pip install --python "${venv_dir}/bin/python" -r "${script_dir}/requirements.lock.txt"
else
  rm -rf "${venv_dir}"
  python3 -m venv "${venv_dir}"
  "${venv_dir}/bin/pip" install --upgrade pip wheel
  "${venv_dir}/bin/pip" install -r "${script_dir}/requirements.lock.txt"
fi

echo "CAD-Recode venv ready at ${venv_dir}"
