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


# 下载失败可以使用其他方式下载并放到在 $build_dir 目录，参考如下命令解压
URL="https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz"
FOLDER="gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu"

aarch64-none-linux-gnu-gcc -v
if [ $? -ne 0 ]; then
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
    aarch64-none-linux-gnu-gcc -v
    if [ $? -ne 0 ]; then
        echo "Error: aarch64-none-linux-gnu-gcc not found"
        exit 1
    fi
fi

opencv_aarch64_url=https://github.com/ZHEQIUSHUI/assets/releases/download/ax650/libopencv-4.5.5-aarch64.zip
if [ ! -f "libopencv-4.5.5-aarch64.zip" ]; then
    # Download the file
    echo "Downloading $opencv_aarch64_url"
    wget "$opencv_aarch64_url" -O "libopencv-4.5.5-aarch64.zip"
else
    echo "libopencv-4.5.5-aarch64.zip already exists"
fi

# Check if the folder exists
if [ ! -d "libopencv-4.5.5-aarch64" ]; then
    # Extract the file
    echo "Extracting unzip libopencv-4.5.5-aarch64.zip"
    unzip libopencv-4.5.5-aarch64.zip
else
    echo "libopencv-4.5.5-aarch64 already exists"
fi


tokenizer_url="https://github.com/ZHEQIUSHUI/hf_tokenizer_ffi/releases/download/v0.1.0/hf-tokenizer-sdk-aarch64.tar.gz"
sha_url="${tokenizer_url}.sha256"

tokenizer_folder="hf-tokenizer-sdk-aarch64"
archive="${tokenizer_folder}.tar.gz"
sha_file="${archive}.sha256"

download_files() {
  echo "Downloading: $tokenizer_url"
  wget "$tokenizer_url" -O "$archive"
  echo "Downloading: $sha_url"
  wget "$sha_url" -O "$sha_file"
}

verify_sha() {
  # .sha256 文件内容形如："<hash>  <filename>"
  # 我们把 filename 固定为当前 archive，避免上游写的文件名不一致导致校验失败
  local expected
  expected="$(awk '{print $1}' "$sha_file")"
  echo "${expected}  ${archive}" | sha256sum -c - >/dev/null 2>&1
}

# 1) 确保文件存在
if [[ ! -f "$archive" || ! -f "$sha_file" ]]; then
  download_files
fi

# 2) 校验，不通过就更新下载再校验一次
if ! verify_sha; then
  echo "SHA256 mismatch. Re-downloading..."
  rm -f "$archive" "$sha_file"
  rm -rf "$tokenizer_folder"
  download_files

  if ! verify_sha; then
    echo "ERROR: SHA256 mismatch after re-download"
    echo "sha file:"
    cat "$sha_file" || true
    exit 1
  fi
fi

echo "SHA256 OK: $archive"

# 3) 解压（只在目录不存在时）
if [ ! -d "$tokenizer_folder" ]; then
  echo "Extracting $archive"
  tar -xf "$archive"
else
  echo "$tokenizer_folder already exists"
fi



# 开始编译
cmake -DBSP_MSP_DIR=${BSP_MSP_DIR} \
-DTOKENIZER_BUILD_TESTS=OFF \
-DHF_TOKENIZER_DIR=$PWD/$tokenizer_folder \
-DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-none-linux-gnu.toolchain.cmake \
-DOpenCV_DIR=$PWD/libopencv-4.5.5-aarch64/lib/cmake/opencv4 ..
make -j16
make install