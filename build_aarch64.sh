#!/bin/bash

# build_dir 修改为自己想要的编译目录名称
build_dir=build_axcl_aarch64
echo "build dir: ${build_dir}"
mkdir ${build_dir}
cd ${build_dir}

URL="https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz"
FOLDER="gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu"

aarch64-none-linux-gnu-gcc -v
if [ $? -ne 0 ]; then
    # Check if the file exists
    if [ ! -f "$FOLDER.tar.xz" ]; then
        # Download the file
        echo "Downloading $URL"
        wget "$URL" -O "$FOLDER.tar.xz"
    else
        echo "$FOLDER.tar.xz already exists"
    fi

    # Check if the folder exists
    if [ ! -d "$FOLDER" ]; then
        # Extract the file
        echo "Extracting $FOLDER.tar.xz"
        tar -xf "$FOLDER.tar.xz"
    else
        echo "$FOLDER already exists"
    fi

    export PATH=$PATH:$PWD/$FOLDER/bin/
    aarch64-none-linux-gnu-gcc -v
    if [ $? -ne 0 ]; then
        echo "Error: aarch64-none-linux-gnu-gcc not found"
        exit 1
    fi
else
    echo "aarch64-none-linux-gnu-gcc already exists"
fi

hf_endpoint=${HF_ENDPOINT:-"https://huggingface.co"}
axcl_dir=axcl_dir
echo "axcl not installed, install it in $axcl_dir, hf_endpoint: $hf_endpoint"
axcl_deb_filename=axcl_host_aarch64_V3.10.2_20251111020143_NO5046.deb
axcl_url=${hf_endpoint}/AXERA-TECH/AXCL/resolve/main/v3.10.2/$axcl_deb_filename
mkdir $axcl_dir
cd $axcl_dir
if [ ! -f $axcl_deb_filename ]; then
    wget $axcl_url
fi
if [ ! -d ./usr ]; then
    dpkg-deb -R ./$axcl_deb_filename .
fi
cd ..


opencv_dir=libopencv-4.5.5-aarch64
opencv_aarch64_url=https://github.com/ZHEQIUSHUI/assets/releases/download/ax650/libopencv-4.5.5-aarch64.zip

# Check if the folder exists
if [ ! -d "libopencv-4.5.5-aarch64" ]; then
    if [ ! -f "libopencv-4.5.5-aarch64.zip" ]; then
        # Download the file
        echo "Downloading $opencv_aarch64_url"
        wget "$opencv_aarch64_url" -O "libopencv-4.5.5-aarch64.zip"
    else
        echo "libopencv-4.5.5-aarch64.zip already exists"
    fi
    # Extract the file
    echo "Extracting unzip libopencv-4.5.5-aarch64.zip"
    unzip libopencv-4.5.5-aarch64.zip
else
    echo "libopencv-4.5.5-aarch64 already exists"
fi

# 开始编译
cmake \
    -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-none-linux-gnu.toolchain.cmake \
    -DTOKENIZER_BUILD_TESTS=OFF \
    -DOpenCV_DIR=$PWD/$opencv_dir/lib/cmake/opencv4 \
    -DAXCL_DIR=$PWD/$axcl_dir/usr ..

make -j16
make install