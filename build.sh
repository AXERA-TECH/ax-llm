#!/bin/bash

# build_dir 修改为自己想要的编译目录名称
build_dir=build
echo "build dir: ${build_dir}"
mkdir ${build_dir}
cd ${build_dir}


opencv_dir=opencv-mobile-4.12.0-ubuntu-2404
opencv_url=https://github.com/nihui/opencv-mobile/releases/download/v34/$opencv_dir.zip

# Check if the folder exists
if [ ! -d "$opencv_dir" ]; then
    if [ ! -f "$opencv_dir.zip" ]; then
        # Download the file
        echo "Downloading $opencv_url"
        wget "$opencv_url" -O "$opencv_dir.zip"
    else
        echo "$opencv_dir.zip already exists"
    fi
    # Extract the file
    echo "Extracting unzip $opencv_dir.zip"
    unzip $opencv_dir.zip
else
    echo "$opencv_dir already exists"
fi


# 开始编译
cmake -DBSP_MSP_DIR=${BSP_MSP_DIR} \
-DTOKENIZER_BUILD_TESTS=OFF \
-DOpenCV_DIR=$PWD/$opencv_dir/lib/cmake/opencv4 ..
make -j16
make install