#!/bin/bash

# build_dir 修改为自己想要的编译目录名称
build_dir=build_x86
echo "build dir: ${build_dir}"
mkdir -p ${build_dir}
cd ${build_dir}

axcl_url="https://github.com/ZHEQIUSHUI/assets/releases/download/ax_3.6.2/axcl_3.6.2_x86.zip"
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
    if [ ! -f "axcl_3.6.2_x86.zip" ]; then
        download_file "${axcl_url}" "axcl_3.6.2_x86.zip"
    fi
    unzip axcl_3.6.2_x86.zip
fi
axcl_dir=${PWD}/axcl_3.6.2

# Optional extra CMake arguments (space-separated), e.g.:
#   AXLLM_CMAKE_ARGS="-DCMAKE_DISABLE_FIND_PACKAGE_OpenCV=TRUE"
extra_cmake_args=()
if [ -n "${AXLLM_CMAKE_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    extra_cmake_args=(${AXLLM_CMAKE_ARGS})
fi

# 开始编译
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install -DAXCL_DIR=${axcl_dir} -DBUILD_AX650=OFF -DBUILD_AXCL=ON "${extra_cmake_args[@]}" ..
make -j16
make install
