#!/bin/bash

# BSP_MSP_DIR 这个变量使用*绝对路径*指定到 SDK 的msp/out目录，如下所示（根据自己的目录修改）
# 绝对路径 绝对路径 绝对路径 

# build_dir 修改为自己想要的编译目录名称
build_dir=build_aarch64
echo "build dir: ${build_dir}"
mkdir -p ${build_dir}
cd ${build_dir}

axcl_url="https://github.com/ZHEQIUSHUI/assets/releases/download/ax_3.6.2/axcl_3.6.2_aarch64.zip"
download_file() {
    local url="$1"
    local out="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 10 --retry-all-errors --retry-delay 2 --connect-timeout 20 --max-time 1200 -o "$out" "$url"
    else
        wget -O "$out" "$url"
    fi
}
if [ ! -d "axcl_3.6.2" ]; then
    echo "Downloading axcl from ${axcl_url}"
    if [ ! -f "axcl_3.6.2_aarch64.zip" ]; then
        download_file "${axcl_url}" "axcl_3.6.2_aarch64.zip"
    fi
    unzip axcl_3.6.2_aarch64.zip
fi
axcl_dir=${PWD}/axcl_3.6.2

# Optional extra CMake arguments (space-separated), e.g.:
#   AXLLM_CMAKE_ARGS="-DCMAKE_DISABLE_FIND_PACKAGE_OpenCV=TRUE"
extra_cmake_args=()
if [ -n "${AXLLM_CMAKE_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    extra_cmake_args=(${AXLLM_CMAKE_ARGS})
fi

# 下载失败可以使用其他方式下载并放到在 $build_dir 目录，参考如下命令解压
URL="https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz"
FOLDER="gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu"

if ! command -v aarch64-none-linux-gnu-gcc >/dev/null 2>&1; then
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

    export PATH=$PATH:$PWD/$FOLDER/bin/
    if ! command -v aarch64-none-linux-gnu-gcc >/dev/null 2>&1; then
        echo "Error: aarch64-none-linux-gnu-gcc not found"
        exit 1
    fi
fi

# 开始编译
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install -DAXCL_DIR=${axcl_dir} -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-none-linux-gnu.toolchain.cmake -DBUILD_AX650=OFF -DBUILD_AXCL=ON "${extra_cmake_args[@]}" ..
make -j16
make install
