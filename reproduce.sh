#!/usr/bin/env bash
set -euo pipefail

case_name="${1:-Case1}"
mode="${2:-wl}"

case_dir="cases/${case_name}"
if [[ ! -d "${case_dir}" ]]; then
  echo "Unknown case: ${case_name}" >&2
  exit 2
fi

case "${mode}" in
  wl|WL|wirelength)
    param_file="${case_dir}/WL-driven.json"
    ;;
  thermal|Thermal|thermal-aware)
    param_file="${case_dir}/Thermal-aware.json"
    ;;
  *)
    echo "Unknown mode: ${mode}" >&2
    exit 2
    ;;
esac

echo "case_dir=${case_dir}"
echo "param_file=${param_file}"
