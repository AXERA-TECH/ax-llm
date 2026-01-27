#!/usr/bin/env bash
set -euo pipefail

mkdir -p build_mingw64
cd build_mingw64

# 4) 配置 & 编译 & 安装
cmake -G "MinGW Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/install" \
  ..

cmake --build . --config Release -j 8
cmake --install . --config Release
