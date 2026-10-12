#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# BSP_MSP_DIR 这个变量使用*绝对路径*指定到 SDK 的msp/out目录，如下所示（根据自己的目录修改）
# 绝对路径 绝对路径 绝对路径 

# build_dir 修改为自己想要的编译目录名称
build_dir="${AXLLM_BUILD_DIR:-build}"
echo "build dir: ${build_dir}"
mkdir -p ${build_dir}
cd ${build_dir}

bsp_url="https://github.com/ZHEQIUSHUI/assets/releases/download/ax_3.6.2/msp_3.6.2.zip"
download_file() {
    local url="$1"
    local out="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 10 --retry-all-errors --retry-delay 2 --connect-timeout 20 --max-time 1200 -o "$out" "$url"
    else
        wget -O "$out" "$url"
    fi
}

resolve_abs_dir() {
    local dir="$1"
    (
        cd "$dir" >/dev/null 2>&1
        pwd
    )
}

reuse_build_dir="${AXLLM_REUSE_BUILD_ASSETS_FROM:-}"
if [ -n "${AXLLM_BSP_MSP_DIR:-}" ]; then
    BSP_MSP_DIR="$(resolve_abs_dir "${AXLLM_BSP_MSP_DIR}")"
elif [ -n "${reuse_build_dir}" ] && [ -d "${reuse_build_dir}/msp_3.6.2/out" ]; then
    BSP_MSP_DIR="$(resolve_abs_dir "${reuse_build_dir}/msp_3.6.2/out")"
else
    if [ ! -d "msp_3.6.2" ]; then
        echo "Downloading bsp from ${bsp_url}"
        if [ ! -f "msp_3.6.2.zip" ]; then
            download_file "${bsp_url}" "msp_3.6.2.zip"
        fi
        unzip -o msp_3.6.2.zip
    fi
    BSP_MSP_DIR="$PWD/msp_3.6.2/out"
fi

echo "bsp dir: ${BSP_MSP_DIR}"
# 下面会简单判断 BSP 路径是否正确
if [ ! -d "${BSP_MSP_DIR}" ]; then
    echo "Error: ${BSP_MSP_DIR} is not a directory"
    exit 1
fi

if [ ! -f "${BSP_MSP_DIR}/lib/libax_sys.so" ]; then
    echo "Error: ${BSP_MSP_DIR}/lib/libax_sys.so is not a file"
    exit 1
fi

# OpenCV is disabled by default for AX650 cross builds to avoid accidentally
# linking a host-architecture installation. AXLLM_CMAKE_ARGS can override it.
extra_cmake_args=(-DCMAKE_DISABLE_FIND_PACKAGE_OpenCV=TRUE)
if [ -n "${AXLLM_CMAKE_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    extra_cmake_args+=( ${AXLLM_CMAKE_ARGS} )
fi

# 下载失败可以使用其他方式下载并放到在 $build_dir 目录，参考如下命令解压
URL="https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz"
FOLDER="gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu"

if ! command -v aarch64-none-linux-gnu-gcc >/dev/null 2>&1; then
    toolchain_bin_dir=""
    if [ -n "${AXLLM_TOOLCHAIN_BIN_DIR:-}" ] && [ -d "${AXLLM_TOOLCHAIN_BIN_DIR}" ]; then
        toolchain_bin_dir="$(resolve_abs_dir "${AXLLM_TOOLCHAIN_BIN_DIR}")"
    elif [ -n "${AXLLM_TOOLCHAIN_ROOT:-}" ] && [ -d "${AXLLM_TOOLCHAIN_ROOT}/bin" ]; then
        toolchain_bin_dir="$(resolve_abs_dir "${AXLLM_TOOLCHAIN_ROOT}/bin")"
    elif [ -n "${reuse_build_dir}" ] && [ -d "${reuse_build_dir}/${FOLDER}/bin" ]; then
        toolchain_bin_dir="$(resolve_abs_dir "${reuse_build_dir}/${FOLDER}/bin")"
    else
        # Check if the file exists
        if [ ! -f "$FOLDER.tar.xz" ]; then
            # Download the file
            echo "Downloading $URL"
            download_file "$URL" "$FOLDER.tar.xz"
        fi

        # Check if the folder exists
        if [ ! -d "$FOLDER" ]; then
            # Extract the file
            echo "Extracting $FOLDER.tar.xz"
            tar -xf "$FOLDER.tar.xz"
        fi
        toolchain_bin_dir="$PWD/$FOLDER/bin"
    fi

    export PATH="${toolchain_bin_dir}:$PATH"
    if ! command -v aarch64-none-linux-gnu-gcc >/dev/null 2>&1; then
        echo "Error: aarch64-none-linux-gnu-gcc not found"
        exit 1
    fi
fi

# Some AX8860 BSP SDKs (for example arm64_glibc) are built against glibc
# 2.34/2.38.  Arm GCC 9.2 only carries a glibc 2.30 sysroot, which makes the
# final link fail with C11-thread symbols such as mtx_lock/thrd_create.
# Fail early with an actionable message instead of an obscure undefined-
# reference linker error.
check_glibc_compatibility() {
    local sdk_engine="${BSP_MSP_DIR}/lib/libax_engine.so"
    local gcc_bin
    local sysroot
    local toolchain_libc
    local required_glibc
    local provided_glibc
    local oldest

    gcc_bin="$(command -v aarch64-none-linux-gnu-gcc)" || return 0
    sysroot="$("${gcc_bin}" -print-sysroot)"
    toolchain_libc="$(find "${sysroot}" -name libc.so.6 -print -quit 2>/dev/null)"
    [ -n "${toolchain_libc}" ] || return 0

    required_glibc="$(readelf --version-info "${sdk_engine}" 2>/dev/null \
        | grep -o 'GLIBC_2\.[0-9]\+' | sort -Vu | tail -n 1)"
    provided_glibc="$(readelf --version-info "${toolchain_libc}" 2>/dev/null \
        | grep -o 'GLIBC_2\.[0-9]\+' | sort -Vu | tail -n 1)"
    [ -n "${required_glibc}" ] && [ -n "${provided_glibc}" ] || return 0

    oldest="$(printf '%s\n' "${required_glibc}" "${provided_glibc}" | sort -V | head -n 1)"
    if [ "${oldest}" = "${provided_glibc}" ] && [ "${required_glibc}" != "${provided_glibc}" ]; then
        echo "Error: AX8860 SDK requires ${required_glibc}, but the selected cross toolchain only provides ${provided_glibc}." >&2
        echo "Use an Arm GNU 13.3+ toolchain, for example:" >&2
        printf '  AXLLM_TOOLCHAIN_ROOT=/path/to/arm-gnu-toolchain-13.3.rel1-x86_64-aarch64-none-linux-gnu \\\n' >&2
        printf '  AXLLM_BUILD_DIR=build_ax8860_glibc238 ./build_ax650.sh\n' >&2
        exit 1
    fi
}

check_glibc_compatibility

# 开始编译
cmake -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=./install \
-DBSP_MSP_DIR=${BSP_MSP_DIR} \
-DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-none-linux-gnu.toolchain.cmake \
-DBUILD_AX650=ON -DBUILD_AXCL=OFF "${extra_cmake_args[@]}" ..
make -j16
make install
