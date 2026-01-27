#!/bin/bash

# build_dir 修改为自己想要的编译目录名称
build_dir=build_650
echo "build dir: ${build_dir}"
mkdir ${build_dir}
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

# 开始编译
cmake -DBSP_MSP_DIR=${BSP_MSP_DIR} \
-DTOKENIZER_BUILD_TESTS=OFF \
-DCMAKE_BUILD_TYPE=Release ..
make -j16
make install
