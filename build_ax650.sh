#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# BSP_MSP_DIR 这个变量使用*绝对路径*指定到 SDK 的msp/out目录，如下所示（根据自己的目录修改）
# 绝对路径 绝对路径 绝对路径 

# build_dir 修改为自己想要的编译目录名称
build_dir=build
echo "build dir: ${build_dir}"
mkdir -p ${build_dir}
cd ${build_dir}

bsp_url="https://github.com/ZHEQIUSHUI/assets/releases/download/ax_3.6.2/msp_3.6.2.zip"
if [ ! -d "msp_3.6.2" ]; then
    echo "Downloading bsp from ${bsp_url}"
    if [ ! -f "msp_3.6.2.zip" ]; then
        wget ${bsp_url}
    fi
    unzip msp_3.6.2.zip
fi

BSP_MSP_DIR=$PWD/msp_3.6.2/out/
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

# 下载失败可以使用其他方式下载并放到在 $build_dir 目录，参考如下命令解压
URL="https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz"
FOLDER="gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu"

if ! command -v aarch64-none-linux-gnu-gcc >/dev/null 2>&1; then
    # Check if the file exists
    if [ ! -f "$FOLDER.tar.xz" ]; then
        # Download the file
        echo "Downloading $URL"
        wget "$URL" -O "$FOLDER.tar.xz"
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
cmake -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=./install \
-DBSP_MSP_DIR=${BSP_MSP_DIR} \
-DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-none-linux-gnu.toolchain.cmake \
-DBUILD_AX650=ON ..
make -j16
make install
