# set cmake_install_prefix path
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

# set(CMAKE_TOOLCHAIN_FILE ${CMAKE_SOURCE_DIR}/toolchains/aarch64-none-linux-gnu.toolchain.cmake)
cmake_minimum_required(VERSION 3.0)
project(LLAMA-AX650-CPP)

add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# 后端选项（可组合）
#
#   BUILD_AX650  ON   → 编译 AX650 片上后端，产出可执行文件 axllm
#   BUILD_AXCL   ON   → 编译 AXCL PCIe 后端，产出可执行文件 axllm_axcl
#
# 三种典型配置：
#   build_650          : -DBUILD_AX650=ON  -DBUILD_AXCL=OFF
#   build (x86 axcl)   : -DBUILD_AX650=OFF -DBUILD_AXCL=ON
#   build_axcl_aarch64 : -DBUILD_AX650=ON  -DBUILD_AXCL=ON
# ---------------------------------------------------------------------------
option(BUILD_AX650 "Build AX650 on-chip backend  (produces: axllm)"       ON)
option(BUILD_AXCL  "Build AXCL PCIe backend       (produces: axllm_axcl)" OFF)

message(STATUS "BUILD_AX650 = ${BUILD_AX650}")
message(STATUS "BUILD_AXCL  = ${BUILD_AXCL}")

if(NOT BUILD_AX650 AND NOT BUILD_AXCL)
    message(FATAL_ERROR "At least one of BUILD_AX650 or BUILD_AXCL must be ON")
endif()

# ---------------------------------------------------------------------------
# SDK 路径
# ---------------------------------------------------------------------------

# AX650 on-chip SDK（仅 BUILD_AX650 时需要）
if(BUILD_AX650)
    if(NOT BSP_MSP_DIR)
        if(EXISTS /soc/lib/libax_engine.so)
            message(STATUS "Detected board, BSP_MSP_DIR = /soc")
            set(BSP_MSP_DIR /soc)
        else()
            set(BSP_MSP_DIR ${CMAKE_SOURCE_DIR}/bsp_msp_out/msp/out)
        endif()
    endif()
    message(STATUS "BSP_MSP_DIR = ${BSP_MSP_DIR}")
endif()

# AXCL PCIe SDK（仅 BUILD_AXCL 时需要）
if(BUILD_AXCL)
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
        message(STATUS "AXCL_DIR = ${AXCL_DIR}")
    endif()
    find_package(OpenMP QUIET)
endif()

# ---------------------------------------------------------------------------
add_subdirectory(third_party/tokenizer.axera)
include_directories(third_party/tokenizer.axera/include)

include(overlook.cmake)

include_directories(src)
include_directories(src/runner)
include_directories(src/runner/utils)

set(OPENAI_API_BUILD_EXAMPLES OFF)
set(OPENAI_API_BUILD_TESTS OFF)
add_subdirectory(third_party/openai-api.cpp)

# ---------------------------------------------------------------------------
# 两个后端共用的源文件
# ---------------------------------------------------------------------------
set(COMMON_SOURCES
    src/runner/utils/memory_utils.cpp
    src/runner/utils/cqdm.cpp
    src/runner/LLM.cpp
)

# ---------------------------------------------------------------------------
# AX650 片上后端  →  axllm
# ---------------------------------------------------------------------------
if(BUILD_AX650)
    add_executable(axllm src/main.cpp
        ${COMMON_SOURCES}
        src/runner/ax_model_runner/ax_model_runner_ax650.cpp
    )
    add_executable(llm_smoke tools/llm_smoke.cpp
        ${COMMON_SOURCES}
        src/runner/ax_model_runner/ax_model_runner_ax650.cpp
    )
    target_include_directories(axllm PRIVATE ${BSP_MSP_DIR}/include)
    target_include_directories(llm_smoke PRIVATE ${BSP_MSP_DIR}/include)
    # link_directories 是全局的，用 target 属性更干净；CMake < 3.13 没有
    # target_link_directories，改用 set_target_properties 设置 LINK_FLAGS 或
    # 直接传绝对路径库目录给 target_link_libraries 的 -L 形式：
    target_link_libraries(axllm
        -L${BSP_MSP_DIR}/lib
        ax_engine ax_interpreter ax_sys
        pthread tokenizer openai_api::server
    )
    target_link_libraries(llm_smoke
        -L${BSP_MSP_DIR}/lib
        ax_engine ax_interpreter ax_sys
        pthread tokenizer openai_api::server
    )
    install(TARGETS axllm DESTINATION bin)
    install(TARGETS llm_smoke DESTINATION bin)
endif()

# ---------------------------------------------------------------------------
# AXCL PCIe 后端  →  axllm_axcl
# ---------------------------------------------------------------------------
if(BUILD_AXCL)
    set(AXCL_LINK_LIBS axcl_rt pthread tokenizer openai_api::server)
    if(OpenMP_CXX_FOUND)
        list(APPEND AXCL_LINK_LIBS OpenMP::OpenMP_CXX)
    endif()

    add_executable(axllm_axcl src/main.cpp
        ${COMMON_SOURCES}
        src/runner/ax_model_runner/ax_model_runner_axcl.cpp
        src/runner/utils/axcl_manager.cpp
    )
    target_include_directories(axllm_axcl PRIVATE ${AXCL_DIR}/include)
    target_compile_definitions(axllm_axcl PRIVATE USE_AXCL)
    target_link_libraries(axllm_axcl
        -L${AXCL_DIR}/lib
        ${AXCL_LINK_LIBS}
    )
    install(TARGETS axllm_axcl DESTINATION bin)
endif()

install(DIRECTORY ${CMAKE_SOURCE_DIR}/scripts/ DESTINATION bin/scripts)
