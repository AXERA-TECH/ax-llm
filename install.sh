#!/usr/bin/env bash
set -Eeuo pipefail

# ax-llm installer
#
# 行为概述（按优先级）：
# 1) 若 /proc/ax_proc/board_id 含 AX650 且本机存在 gcc → 下载 BSP 与交叉工具链，编译 AX650 片上后端，安装 axllm 到 /usr/bin
# 2) 否则若本机可运行 axcl-smi 且 /usr/include/axcl 和 /usr/lib/axcl 存在 → 编译 AXCL 后端，安装 axllm_axcl 到 /usr/bin
# 3) 否则退出并提示依赖不足
#
# 额外说明：
# - “下载当前分支的代码”：默认从当前仓库的 origin 克隆并固定使用 axllm 分支；
#   也可通过环境变量覆盖：REPO_URL、BRANCH（若设置则使用该分支）。
# - 安装到 /usr/bin 需要 root 权限，脚本会自动通过 sudo 提权（若存在）。

PROJECT_NAME="ax-llm"
AX650_BIN="axllm"
AXCL_BIN="axllm"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Error: required command '$1' not found" >&2
    exit 1
  }
}

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

get_repo_and_branch() {
  local here
  here=$(cd -- "$(dirname -- "$0")" && pwd -P)

  : "${REPO_URL:=$(git -C "$here" config --get remote.origin.url 2>/dev/null || true)}"
  if [ -z "${REPO_URL}" ] && [ ! -f "${here}/CMakeLists.txt" ]; then
    # fallback for curl | bash usage
    REPO_URL="https://github.com/AXERA-TECH/ax-llm.git"
  fi
  # 强制默认分支为 axllm，允许通过环境变量 BRANCH 覆盖
  : "${BRANCH:=axllm}"
}

clone_current_branch() {
  local workdir
  workdir=$(mktemp -d -t ${PROJECT_NAME}-build-XXXXXX)
  echo "Working dir: ${workdir}"

  if [ -n "${REPO_URL:-}" ]; then
    need_cmd git
    echo "Cloning ${REPO_URL} (branch ${BRANCH}) ..."
    if git ls-remote --exit-code --heads "${REPO_URL}" "${BRANCH}" >/dev/null 2>&1; then
      git clone --recurse-submodules --depth 1 --branch "${BRANCH}" "${REPO_URL}" "${workdir}/${PROJECT_NAME}"
    else
      echo "Warning: branch '${BRANCH}' not found on remote; cloning default branch and then trying checkout."
      git clone --recurse-submodules --depth 1 "${REPO_URL}" "${workdir}/${PROJECT_NAME}"
      (cd "${workdir}/${PROJECT_NAME}" && git fetch --depth 1 origin "${BRANCH}" || true && git checkout -q "${BRANCH}" || true)
    fi
    SRC_DIR="${workdir}/${PROJECT_NAME}"
  else
    # 没有可用的远端，直接在当前仓库内构建
    echo "No REPO_URL detected; building in-place."
    SRC_DIR=$(cd -- "$(dirname -- "$0")" && pwd -P)
    if [ ! -f "${SRC_DIR}/CMakeLists.txt" ]; then
      echo "Error: no repository found and CMakeLists.txt missing (curl | bash requires REPO_URL)." >&2
      exit 1
    fi
    workdir="$SRC_DIR"  # for trap cleanup logic
  fi

  # 更新子模块（如有）
  if [ -f "${SRC_DIR}/.gitmodules" ]; then
    if command -v git >/dev/null 2>&1; then
      echo "Updating git submodules..."
      git -C "${SRC_DIR}" submodule sync --recursive || true
      git -C "${SRC_DIR}" submodule update --init --recursive
    else
      echo "Warning: .gitmodules found but git not available; skipping submodules." >&2
    fi
  fi
  # 清理策略：若使用了 mktemp 目录则退出时清理
  if [[ "$workdir" == /tmp/${PROJECT_NAME}-build-* ]]; then
    CLEANUP_DIR="$workdir"
    trap 'rm -rf -- "$CLEANUP_DIR"' EXIT
  fi
}

fetch_ax650_bsp_and_toolchain() {
  # 参考 build_ax650.sh 的 BSP 下载逻辑（本分支不下载工具链）
  if ! command -v gcc >/dev/null 2>&1; then
    echo "Error: gcc not found; AX650 build requires local gcc." >&2
    exit 1
  fi
  local build_dir=buiLd_ax650_auto
  mkdir -p "$build_dir" && cd "$build_dir"

  local bsp_url="https://github.com/ZHEQIUSHUI/assets/releases/download/ax_3.6.2/msp_3.6.2.zip"
  local bsp_cache_dir="/tmp/ax-llm-bsp"
  local bsp_zip="${bsp_cache_dir}/msp_3.6.2.zip"
  local bsp_root="${bsp_cache_dir}/msp_3.6.2"
  local bsp_out="${bsp_root}/out"

  mkdir -p "${bsp_cache_dir}"
  if [ -d "${bsp_out}" ] && [ -f "${bsp_out}/lib/libax_sys.so" ]; then
    echo "Using cached BSP at ${bsp_out}"
  else
    echo "Downloading BSP from ${bsp_url}"
    need_cmd wget
    need_cmd unzip
    [ -f "${bsp_zip}" ] || wget -q --show-progress "$bsp_url" -O "${bsp_zip}"
    rm -rf "${bsp_root}"
    unzip -q "${bsp_zip}" -d "${bsp_cache_dir}"
  fi

  BSP_MSP_DIR="${bsp_out}"
  AX650_BUILD_DIR="$PWD"
}

build_ax650() {
  echo "==> Building AX650 backend (axllm)"
  cd "$SRC_DIR"
  fetch_ax650_bsp_and_toolchain

  cmake -S .. -B . \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DBSP_MSP_DIR="${BSP_MSP_DIR}" \
    -DBUILD_AX650=ON \
    -DBUILD_AXCL=OFF
  make -j"${JOBS}"
  make install
  as_root install -Dm755 install/bin/${AX650_BIN} /usr/bin/${AX650_BIN}
  echo "Installed /usr/bin/${AX650_BIN}"
}

build_axcl_system() {
  echo "==> Building AXCL backend (axllm) using system /usr/include/axcl and /usr/lib/axcl"
  cd "$SRC_DIR"
  local build_dir=build_axcl_sys
  mkdir -p "$build_dir" && cd "$build_dir"
  cmake -S .. -B . \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DBUILD_AX650=OFF \
    -DBUILD_AXCL=ON
  make -j"${JOBS}"
  make install
  as_root install -Dm755 install/bin/${AXCL_BIN} /usr/bin/${AXCL_BIN}
  echo "Installed /usr/bin/${AXCL_BIN}"
}

main() {
  # 通用依赖
  need_cmd cmake
  need_cmd make
  JOBS=${JOBS:-$(command -v nproc >/dev/null 2>&1 && nproc || echo 8)}

  get_repo_and_branch
  clone_current_branch

  local is_ax650=false
  if [ -r /proc/ax_proc/board_id ] && grep -q "AX650" /proc/ax_proc/board_id 2>/dev/null; then
    if command -v gcc >/dev/null 2>&1; then
      is_ax650=true
    else
      echo "Detected AX650 board, but gcc not found; skipping AX650 path."
    fi
  fi

  if $is_ax650; then
    build_ax650
    exit 0
  fi

  # Fallback: AXCL 本地环境（工具 + 头文件 + 库）
  if command -v axcl-smi >/dev/null 2>&1 \
     && [ -d /usr/include/axcl ] \
     && [ -d /usr/lib/axcl ]; then
    build_axcl_system
    exit 0
  fi

  echo "Error: neither AX650 (with gcc) nor AXCL SDK detected." >&2
  echo "- AX650 path requires: /proc/ax_proc/board_id contains AX650 AND gcc present" >&2
  echo "- AXCL path requires: axcl-smi available AND /usr/include/axcl + /usr/lib/axcl present" >&2
  echo "You may also set REPO_URL and BRANCH to control clone source." >&2
  exit 2
}

main "$@"
