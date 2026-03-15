#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
source "${script_dir}/pins.env"

clone_or_update() {
  local name="$1"
  local url="$2"
  local ref="$3"
  local dest="${script_dir}/${name}"

  if [[ ! -d "${dest}/.git" ]]; then
    git clone "${url}" "${dest}"
  fi

  git -C "${dest}" fetch --tags origin
  git -C "${dest}" checkout "${ref}"
}

clone_or_update "cad-recode" "${CAD_RECODE_REPO}" "${CAD_RECODE_REF}"
clone_or_update "cadrille" "${CADRILLE_REPO}" "${CADRILLE_REF}"
clone_or_update "point2cad" "${POINT2CAD_REPO}" "${POINT2CAD_REF}"

echo "Pinned upstream repos are ready under ${script_dir}"
