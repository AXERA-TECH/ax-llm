#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Compatibility entrypoint: default to AX650 board build.
exec "${SCRIPT_DIR}/build_ax650.sh" "$@"
