#!/usr/bin/env bash
set -euo pipefail

build_dir="build"
tag="axllm-ax650:local"
output=""

usage() {
  cat <<EOF
Usage: $0 [--build-dir DIR] [--tag TAG] [--output image.tar.gz]

Build an AX650 Docker image from local build outputs.

Defaults:
  --build-dir build
  --tag       axllm-ax650:local

Prerequisites:
  - Build outputs exist (run ./build_ax650.sh first)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir) build_dir="$2"; shift 2 ;;
    --tag) tag="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

bin_path="${build_dir}/install/bin/axllm"
if [[ ! -f "$bin_path" ]]; then
  # Compatibility: some build dirs may place the binary directly under build_dir/
  if [[ -f "${build_dir}/axllm" ]]; then
    bin_path="${build_dir}/axllm"
  else
    echo "Missing axllm binary: ${build_dir}/install/bin/axllm (or ${build_dir}/axllm)" >&2
    exit 1
  fi
fi

if command -v readelf >/dev/null 2>&1; then
  if readelf -d "$bin_path" 2>/dev/null | grep -q "Shared library: \\[libopencv"; then
    echo "Error: $bin_path links against OpenCV; rebuild with:" >&2
    echo "  AXLLM_CMAKE_ARGS=\"-DCMAKE_DISABLE_FIND_PACKAGE_OpenCV=TRUE\" ./build_ax650.sh" >&2
    exit 1
  fi
fi

ctx="$(mktemp -d)"
cleanup() { rm -rf "$ctx"; }
trap cleanup EXIT

cp -a "$bin_path" "$ctx/axllm"
cp -a docker/ax650/Dockerfile "$ctx/Dockerfile"

# AX650 runs on aarch64 boards; build an arm64 image.
docker buildx build --builder default --platform linux/arm64 -t "$tag" --load "$ctx"

if [[ -n "$output" ]]; then
  mkdir -p "$(dirname "$output")"
  docker save "$tag" | gzip -c > "$output"
  echo "Saved: $output"
fi

echo "Built image: $tag"
