#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case_name="${1:-Case1}"
mode="${2:-wl}"

case_dir="${script_dir}/cases/${case_name}"
if [[ ! -d "${case_dir}" ]]; then
  echo "Unknown case: ${case_name}" >&2
  exit 2
fi

case "${mode}" in
  wl|WL|wirelength)
    mode="wl"
    param_file="${case_dir}/WL-driven.json"
    ;;
  thermal|Thermal|thermal-aware)
    mode="thermal"
    param_file="${case_dir}/Thermal-aware.json"
    ;;
  *)
    echo "Unknown mode: ${mode}" >&2
    exit 2
    ;;
esac

if [[ ! -f "${param_file}" ]]; then
  echo "Missing parameter file: ${param_file}" >&2
  exit 2
fi

python_bin="${PYTHON:-python3}"
stamp="$(date +%Y%m%d_%H%M%S)"
out_dir="${ATPLACE_OUT_DIR:-${script_dir}/results/${case_name}_${mode}_${stamp}}"

if [[ -z "${USERNAME:-}" ]]; then
  USERNAME="${USER:-}"
fi
if [[ -z "${USERNAME}" ]]; then
  USERNAME="$(id -un)"
fi
export USERNAME
export USER="${USER:-${USERNAME}}"
export LOGNAME="${LOGNAME:-${USERNAME}}"
export LC_ALL=C

echo "case_dir=${case_dir}"
echo "param_file=${param_file}"
echo "out_dir=${out_dir}"
echo "python_entry=${script_dir}/reproduce.py"

if [[ "${ATPLACE_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

exec "${python_bin}" "${script_dir}/reproduce.py" \
  --case "${case_name}" \
  --mode "${mode}" \
  --case-dir "${case_dir}" \
  --param-file "${param_file}" \
  --out-dir "${out_dir}"
