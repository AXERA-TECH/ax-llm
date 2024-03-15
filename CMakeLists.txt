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

# 添加 FLAGS 检查代码是否有明显 bug
# include(cmake/overlook.cmake)
include(overlook.cmake)

add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# bsp
if(NOT BSP_MSP_DIR)
    set(BSP_MSP_DIR ${CMAKE_SOURCE_DIR}/bsp_msp_out/msp/out)
endif()
message(STATUS "BSP_MSP_DIR = ${BSP_MSP_DIR}")
include_directories(${BSP_MSP_DIR}/include)
link_directories(${BSP_MSP_DIR}/lib)

set(SPM_ENABLE_SHARED OFF)
add_subdirectory(third_party/sentencepiece)
include_directories(third_party/sentencepiece/src)
include_directories(third_party/sentencepiece/third_party/protobuf-lite)

set(ABSL_ENABLE_INSTALL ON)
set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory(third_party/abseil-cpp)
add_subdirectory(third_party/re2)
include_directories(third_party/abseil-cpp)
include_directories(third_party/re2)
link_directories(${CMAKE_BINARY_DIR}/lib)

include_directories(src)
include_directories(src/runner)
include_directories(src/runner/utils)

add_executable(main src/main.cpp 
                src/runner/ax_model_runner/ax_model_runner_ax650.cpp 
                src/runner/utils/memory_utils.cpp 
                src/runner/utils/cqdm.cpp
                src/runner/Tokenizer/Tokenizer.cpp
                src/runner/Tokenizer/QwenTokenizer.cpp)

target_link_libraries(main ax_engine ax_interpreter ax_sys)
target_link_libraries(main sentencepiece re2::re2)

install(TARGETS main DESTINATION bin)
file(GLOB RUN_SCRIPT "${CMAKE_SOURCE_DIR}/scripts/*")
install(FILES ${RUN_SCRIPT} DESTINATION bin/)

# add_executable(fp32_to_bf16 tools/fp32_to_bf16.cpp)
