#!/usr/bin/env bash
set -euo pipefail

arch="amd64"
build_dir=""
tag="axllm-axcl:local"
output=""

usage() {
  cat <<EOF
Usage: $0 [--arch amd64|arm64] [--build-dir DIR] [--tag TAG] [--output image.tar.gz]

Build an AXCL Docker image from local build outputs.

Defaults:
  --arch      amd64
  --build-dir build_x86 (amd64) / build_aarch64 (arm64)
  --tag       axllm-axcl:local

Prerequisites:
  - Build outputs exist (run ./build_axcl_x86.sh or ./build_axcl_aarch64.sh first)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch) arch="$2"; shift 2 ;;
    --build-dir) build_dir="$2"; shift 2 ;;
    --tag) tag="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

case "$arch" in
  amd64) : ;;
  arm64) : ;;
  *) echo "Invalid --arch: $arch" >&2; exit 2 ;;
esac

if [[ -z "$build_dir" ]]; then
  if [[ "$arch" == "amd64" ]]; then
    build_dir="build_x86"
  else
    build_dir="build_aarch64"
  fi
fi

bin_path="${build_dir}/install/bin/axllm"
sdk_lib_dir="${build_dir}/axcl_3.6.2/lib/axcl"

if [[ ! -f "$bin_path" ]]; then
  echo "Missing axllm binary: $bin_path" >&2
  exit 1
fi
if [[ ! -d "$sdk_lib_dir" ]]; then
  echo "Missing AXCL SDK libs: $sdk_lib_dir" >&2
  exit 1
fi

if command -v readelf >/dev/null 2>&1; then
  if readelf -d "$bin_path" 2>/dev/null | grep -q "Shared library: \\[libopencv"; then
    echo "Error: $bin_path links against OpenCV; rebuild with:" >&2
    echo "  AXLLM_CMAKE_ARGS=\"-DCMAKE_DISABLE_FIND_PACKAGE_OpenCV=TRUE\" ./build_axcl_x86.sh" >&2
    exit 1
  fi
fi

ctx="$(mktemp -d)"
cleanup() { rm -rf "$ctx"; }
trap cleanup EXIT

mkdir -p "$ctx/axcl/lib"
cp -a "$bin_path" "$ctx/axllm"
cp -a "${sdk_lib_dir}/." "$ctx/axcl/lib/"
cp -a docker/axcl/Dockerfile "$ctx/Dockerfile"

if [[ "$arch" == "arm64" ]]; then
  docker buildx build --builder default --platform linux/arm64 -t "$tag" --load "$ctx"
else
  docker build -t "$tag" "$ctx"
fi

if [[ -n "$output" ]]; then
  mkdir -p "$(dirname "$output")"
  docker save "$tag" | gzip -c > "$output"
  echo "Saved: $output"
fi

echo "Built image: $tag"
