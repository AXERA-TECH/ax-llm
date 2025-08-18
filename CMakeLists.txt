# set cmake_install_prefix path
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()

message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

# check if building type is not set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

# set(CMAKE_TOOLCHAIN_FILE ${CMAKE_SOURCE_DIR}/toolchains/aarch64-none-linux-gnu.toolchain.cmake)
cmake_minimum_required(VERSION 3.0)
project(LLAMA-AX650-CPP)

add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_WITH_YOLOV5_TEST OFF)

# set(SPM_ENABLE_SHARED OFF)
# add_subdirectory(third_party/sentencepiece)
# include_directories(third_party/sentencepiece/src)
# include_directories(third_party/sentencepiece/third_party/protobuf-lite)

# set(ABSL_ENABLE_INSTALL ON)
# set(ABSL_PROPAGATE_CXX_STD ON)
# add_subdirectory(third_party/abseil-cpp)
# add_subdirectory(third_party/re2)
# include_directories(third_party/abseil-cpp)
# include_directories(third_party/re2)
# link_directories(${CMAKE_BINARY_DIR}/lib)

# 添加 FLAGS 检查代码是否有明显 bug
include(overlook.cmake)

find_package(OpenMP REQUIRED)

include_directories(src)
include_directories(src/runner)
include_directories(src/runner/utils)

if (NOT AXCL_DIR)
    include_directories(/usr/include/axcl/)
    link_directories(/usr/lib/axcl/)
else()
    include_directories(${AXCL_DIR}/include)
    include_directories(${AXCL_DIR}/include/include)
    include_directories(${AXCL_DIR}/include/axcl/)
    include_directories(${AXCL_DIR}/include/axcl/include)
    link_directories(${AXCL_DIR}/lib)
    link_directories(${AXCL_DIR}/lib/axcl)
endif()

function(build_exec name main_source)
    add_executable(${name} ${main_source}
                    src/runner/ax_model_runner/ax_model_runner_ax650.cpp 
                    # src/runner/ax_model_runner/ax_model_runner_ax650_host.cpp
                    src/runner/utils/axcl_manager.cpp
                    src/runner/utils/memory_utils.cpp 
                    src/runner/utils/cqdm.cpp
                    src/runner/Tokenizer/Tokenizer.cpp
                    # src/runner/Tokenizer/QwenTokenizer.cpp
                    )

    # target_link_libraries(${name} ax_engine ax_interpreter ax_sys)
    
    target_link_libraries(${name} spdlog axcl_pcie_dma axcl_pcie_msg axcl_token axcl_comm axcl_pkg axcl_rt pthread OpenMP::OpenMP_CXX)

    # target_link_libraries(${name} sentencepiece re2::re2)
    install(TARGETS ${name} DESTINATION bin)
endfunction()

build_exec(main src/main.cpp)
build_exec(main_api src/main_api.cpp)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/scripts/ DESTINATION bin/scripts)