#!/bin/bash

# build_dir 修改为自己想要的编译目录名称
build_dir=build_axcl_x86
echo "build dir: ${build_dir}"
mkdir ${build_dir}
cd ${build_dir}

if [ ! -d /usr/include/axcl/ ]; then
    hf_endpoint=${HF_ENDPOINT:-"https://huggingface.co"}
    axcl_dir=axcl_dir
    echo "axcl not installed, install it in $axcl_dir, hf_endpoint: $hf_endpoint"
    axcl_deb_filename=axcl_host_x86_64_V3.10.2_20251111020143_NO5046.deb
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
fi


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
if [ ! -d /usr/include/axcl/ ]; then
    cmake \
        -DTOKENIZER_BUILD_TESTS=OFF \
        -DOpenCV_DIR=$PWD/$opencv_dir/lib/cmake/opencv4 \
        -DAXCL_DIR=$PWD/$axcl_dir/usr \
        .. \
	    -DCMAKE_BUILD_TYPE=Release
else
    cmake \
        -DTOKENIZER_BUILD_TESTS=OFF \
        -DOpenCV_DIR=$PWD/$opencv_dir/lib/cmake/opencv4 .. \
	    -DCMAKE_BUILD_TYPE=Release
fi
make -j16
make install
