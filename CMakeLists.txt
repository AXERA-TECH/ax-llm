# set cmake_install_prefix path
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()

message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

# check if building type is not set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

# bsp
if(NOT BSP_MSP_DIR)
    set(BSP_MSP_DIR ${CMAKE_SOURCE_DIR}/bsp_msp_out/msp/out)
endif()

message(STATUS "BSP_MSP_DIR = ${BSP_MSP_DIR}")

# set(CMAKE_TOOLCHAIN_FILE ${CMAKE_SOURCE_DIR}/toolchains/aarch64-none-linux-gnu.toolchain.cmake)
cmake_minimum_required(VERSION 3.0)
project(LLAMA-AX650-CPP)

include(overlook.cmake)

set(SPM_ENABLE_SHARED OFF)
add_subdirectory(third_party/sentencepiece)

# 添加 FLAGS 检查代码是否有明显 bug
# include(cmake/overlook.cmake)

add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${BSP_MSP_DIR}/include)
link_directories(${BSP_MSP_DIR}/lib)

include_directories(third_party/sentencepiece/src)
include_directories(third_party/sentencepiece/third_party/protobuf-lite)

# add_library(owlvit STATIC
#     src/runner/ax_model_runner_ax650.cpp
# )

add_executable(main src/main.cpp src/runner/ax_model_runner_ax650.cpp src/runner/memory_utils.cpp src/runner/cqdm.cpp)

target_link_libraries(main ax_engine ax_interpreter ax_sys)

target_link_libraries(main sentencepiece)

install(TARGETS main DESTINATION bin)
file(GLOB RUN_SCRIPT "${CMAKE_SOURCE_DIR}/scripts/*")
install(FILES ${RUN_SCRIPT} DESTINATION bin/)

add_executable(fp32_to_bf16 tools/fp32_to_bf16.cpp)
