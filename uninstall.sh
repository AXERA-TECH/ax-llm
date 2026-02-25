#!/usr/bin/env bash
set -Eeuo pipefail

as_root() {
  if [ "${EUID:-$(id -u)}" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
      sudo "$@"
    else
      echo "Error: need root privileges to run: $*" >&2
      echo "Hint: rerun as root or install sudo." >&2
      exit 1
    fi
  else
    "$@"
  fi
}

BINS=(axllm axllm_axcl)

removed_any=false
for b in "${BINS[@]}"; do
  if [ -e "/usr/bin/$b" ]; then
    echo "Removing /usr/bin/$b"
    as_root rm -f "/usr/bin/$b"
    removed_any=true
  fi
done

if ! $removed_any; then
  echo "Nothing to remove under /usr/bin (axllm, axllm_axcl not found)."
fi

