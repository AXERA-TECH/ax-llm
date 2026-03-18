#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_FILE="${TMPDIR:-/tmp}/axllm_windows_axcl.patch"

REMOTE_HOST="${REMOTE_HOST:-i@10.126.35.160}"
REMOTE_PASS="${REMOTE_PASS:-0}"
REMOTE_REPO="${REMOTE_REPO:-C:\\Users\\i\\projetcs\\ax-llm}"
REMOTE_BUILD_DIR="${REMOTE_BUILD_DIR:-build_windows_mingw}"
REMOTE_AXCL_DIR="${REMOTE_AXCL_DIR:-C:\\Program Files\\AXCL\\axcl\\out\\axcl_win_x64}"

FILES=(
  CMakeLists.txt
  overlook.cmake
  src/main.cpp
  src/runner/utils/files.cpp
  src/runner/utils/memory_utils.hpp
  src/runner/vision/vision_module.cpp
  src/runner/utils/cqdm.h
  src/runner/utils/cqdm.cpp
  src/runner/ax_model_runner/ax_model_runner_axcl.cpp
)

SSH_BASE=(
  sshpass -p "${REMOTE_PASS}"
  ssh
  -o PreferredAuthentications=password
  -o StrictHostKeyChecking=no
  "${REMOTE_HOST}"
)

SCP_BASE=(
  sshpass -p "${REMOTE_PASS}"
  scp
  -o PreferredAuthentications=password
  -o StrictHostKeyChecking=no
)

build_patch() {
  (cd "${ROOT_DIR}" && git diff -- "${FILES[@]}" > "${PATCH_FILE}")
}

remote_cmd() {
  "${SSH_BASE[@]}" "$1"
}

sync_patch() {
  "${SCP_BASE[@]}" "${PATCH_FILE}" "${REMOTE_HOST}:/C:/Users/i/projetcs/ax-llm/axllm_windows_axcl.patch"
}

apply_patch_remote() {
  local repo="${REMOTE_REPO}"
  remote_cmd "cd /d ${repo} && git apply --check axllm_windows_axcl.patch"
}

reapply_patch_remote() {
  local repo="${REMOTE_REPO}"
  if ! remote_cmd "cd /d ${repo} && git apply --check axllm_windows_axcl.patch"; then
    remote_cmd "cd /d ${repo} && git apply -R axllm_windows_axcl.patch"
    remote_cmd "cd /d ${repo} && git apply --check axllm_windows_axcl.patch"
  fi
  remote_cmd "cd /d ${repo} && git apply axllm_windows_axcl.patch"
}

configure_and_build() {
  local repo="${REMOTE_REPO}"
  local build_dir="${REMOTE_BUILD_DIR}"
  local axcl_dir="${REMOTE_AXCL_DIR}"

  remote_cmd "cd /d ${repo} && if exist ${build_dir} rmdir /s /q ${build_dir}"
  remote_cmd "cd /d ${repo} && mkdir ${build_dir}"
  remote_cmd "cd /d ${repo}\\${build_dir} && cmake -G \"MinGW Makefiles\" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install -DAXCL_DIR=\"${axcl_dir}\" -DBUILD_AX650=OFF -DBUILD_AXCL=ON .."
  remote_cmd "cd /d ${repo}\\${build_dir} && cmake --build . --parallel 8"
}

main() {
  build_patch
  sync_patch
  reapply_patch_remote
  configure_and_build
}

main "$@"
