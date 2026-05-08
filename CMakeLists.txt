# set cmake_install_prefix path
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

# set(CMAKE_TOOLCHAIN_FILE ${CMAKE_SOURCE_DIR}/toolchains/aarch64-none-linux-gnu.toolchain.cmake)
cmake_minimum_required(VERSION 3.10)
project(AX-LLM-CPP)

if(NOT MSVC)
    add_compile_options(-std=c++17)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(WIN32)
    # Ensure cpp-httplib (used by openai-api.cpp) sees a modern Windows SDK target.
    # Without this, MinGW may default to an older _WIN32_WINNT and fail compilation.
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _WIN32_WINNT=0x0A00 WINVER=0x0A00)
endif()

# ---------------------------------------------------------------------------
# 后端选项（可组合）
#
#   BUILD_AX650  ON   → 编译 AX650 片上后端，产出可执行文件 axllm
#   BUILD_AXCL   ON   → 编译 AXCL PCIe 后端，产出可执行文件 axllm
#
# 三种典型配置：
#   build_650          : -DBUILD_AX650=ON  -DBUILD_AXCL=OFF
#   build (x86 axcl)   : -DBUILD_AX650=OFF -DBUILD_AXCL=ON
#   build_axcl_aarch64 : -DBUILD_AX650=ON  -DBUILD_AXCL=ON
# ---------------------------------------------------------------------------
option(BUILD_AX650 "Build AX650 on-chip backend  (produces: axllm)"       ON)
option(BUILD_AXCL  "Build AXCL PCIe backend       (produces: axllm)" OFF)

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
    if(NOT AXCL_DIR AND DEFINED ENV{AXCL_DIR})
        set(AXCL_DIR "$ENV{AXCL_DIR}")
    endif()

    if(WIN32 AND NOT AXCL_DIR)
        message(FATAL_ERROR "BUILD_AXCL on Windows requires -DAXCL_DIR=<AXCL SDK root>")
    endif()

    set(AXCL_INCLUDE_DIRS "")
    set(AXCL_LIBRARY_DIRS "")
    if (NOT AXCL_DIR)
        list(APPEND AXCL_INCLUDE_DIRS /usr/include/axcl/)
        list(APPEND AXCL_LIBRARY_DIRS /usr/lib/axcl/)
    else()
        list(APPEND AXCL_INCLUDE_DIRS
            ${AXCL_DIR}/include
            ${AXCL_DIR}/include/include
            ${AXCL_DIR}/include/axcl/
            ${AXCL_DIR}/include/axcl/include
        )
        list(APPEND AXCL_LIBRARY_DIRS
            ${AXCL_DIR}/lib
            ${AXCL_DIR}/lib/axcl
            ${AXCL_DIR}/bin
        )
        message(STATUS "AXCL_DIR = ${AXCL_DIR}")
    endif()
    include_directories(${AXCL_INCLUDE_DIRS})
    if(AXCL_LIBRARY_DIRS)
        link_directories(${AXCL_LIBRARY_DIRS})
    endif()
    find_package(OpenMP QUIET)
    find_package(Threads REQUIRED)
endif()

# OpenMP for AX650 on-chip path (per-layer input compute uses #pragma omp parallel for)
if(BUILD_AX650 AND NOT OpenMP_CXX_FOUND)
    find_package(OpenMP QUIET)
endif()

# ---------------------------------------------------------------------------
add_subdirectory(third_party/tokenizer.axera)
include_directories(third_party/tokenizer.axera/include)
include_directories(third_party/magic_enum/include)

include(overlook.cmake)

include_directories(src)
include_directories(src/runner)
include_directories(src/runner/utils)

# ---------------------------------------------------------------------------
# Vision preprocessing (OpenCV)
# ---------------------------------------------------------------------------
set(AXLLM_VISION_LIBS "")
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found (${OpenCV_VERSION})")
    add_compile_definitions(AXLLM_USE_OPENCV)
    include_directories(${OpenCV_INCLUDE_DIRS})
    set(AXLLM_VISION_LIBS ${OpenCV_LIBS})
else()
    # Fallback: SimpleCV (stb-based tiny OpenCV-like lib)
    if(EXISTS ${CMAKE_SOURCE_DIR}/third_party/SimpleCV/CMakeLists.txt)
        message(WARNING "OpenCV not found: using third_party/SimpleCV for image preprocessing (may have slight differences vs OpenCV)")
        set(SIMPLECV_BUILD_TESTS OFF CACHE BOOL "Build SimpleCV tests" FORCE)
        add_subdirectory(third_party/SimpleCV)
        set(AXLLM_VISION_LIBS SimpleCV::simplecv)
    else()
        message(FATAL_ERROR "OpenCV not found and third_party/SimpleCV is missing. Install OpenCV or init submodule third_party/SimpleCV.")
    endif()
endif()

set(OPENAI_API_BUILD_EXAMPLES OFF)
set(OPENAI_API_BUILD_TESTS OFF)
set(OPENAI_API_BUILD_SERVER OFF)
add_subdirectory(third_party/openai-api.cpp)

# ---------------------------------------------------------------------------
# 两个后端共用的源文件
# ---------------------------------------------------------------------------
set(COMMON_SOURCES
    src/runner/utils/memory_utils.cpp
    src/runner/utils/audio_processor.cpp
    src/runner/utils/cqdm.cpp
    src/runner/utils/logger.cpp
    src/runner/utils/net_utils.cpp
    src/runner/LLM.cpp
    src/runner/utils/files.cpp
    src/runner/utils/image_processor.cpp
    src/runner/utils/mrope.cpp
    src/runner/vision/vision_module.cpp
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
        ${AXLLM_VISION_LIBS}
    )
    target_link_libraries(llm_smoke
        -L${BSP_MSP_DIR}/lib
        ax_engine ax_interpreter ax_sys
        pthread tokenizer openai_api::server
        ${AXLLM_VISION_LIBS}
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(axllm OpenMP::OpenMP_CXX)
        target_link_libraries(llm_smoke OpenMP::OpenMP_CXX)
    endif()
    install(TARGETS axllm DESTINATION bin)
    install(TARGETS llm_smoke DESTINATION bin)
endif()

# ---------------------------------------------------------------------------
# AXCL PCIe 后端  →  axllm
# ---------------------------------------------------------------------------
if(BUILD_AXCL)
    set(AXCL_LINK_LIBS tokenizer openai_api::server Threads::Threads)

    if(WIN32)
        find_library(AXCL_RT_LIBRARY
            NAMES axcl_rt libaxcl_rt
            PATHS ${AXCL_LIBRARY_DIRS}
        )
        if(NOT AXCL_RT_LIBRARY)
            message(FATAL_ERROR "Failed to locate AXCL runtime library (axcl_rt) under AXCL_DIR=${AXCL_DIR}")
        endif()
        list(PREPEND AXCL_LINK_LIBS ${AXCL_RT_LIBRARY})
    else()
        set(AXCL_LINK_LIBS axcl_rt ${AXCL_LINK_LIBS})

        # Prefer real versioned axcl_rt .so to avoid linker script extra deps
        file(GLOB AXCL_RT_CAND "${AXCL_DIR}/lib/libaxcl_rt.so.*" "${AXCL_DIR}/lib/axcl/libaxcl_rt.so.*")
        if(AXCL_RT_CAND)
            list(SORT AXCL_RT_CAND COMPARE NATURAL ORDER DESCENDING)
            list(GET AXCL_RT_CAND 0 AXCL_RT_REAL)
            add_library(axcl_rt_real SHARED IMPORTED)
            set_target_properties(axcl_rt_real PROPERTIES IMPORTED_LOCATION "${AXCL_RT_REAL}")
            set(AXCL_LINK_LIBS axcl_rt_real tokenizer openai_api::server Threads::Threads)
        endif()
    endif()

    if(OpenMP_CXX_FOUND)
        list(APPEND AXCL_LINK_LIBS OpenMP::OpenMP_CXX)
    endif()

    add_executable(axllm src/main.cpp
        ${COMMON_SOURCES}
        src/runner/ax_model_runner/ax_model_runner_axcl.cpp
        src/runner/utils/axcl_manager.cpp
    )
    target_include_directories(axllm PRIVATE ${AXCL_INCLUDE_DIRS})
    target_compile_definitions(axllm PRIVATE USE_AXCL)
    target_link_libraries(axllm
        ${AXCL_LINK_LIBS}
        ${AXLLM_VISION_LIBS}
    )
    # AXCL smoke test
    add_executable(llm_smoke tools/llm_smoke.cpp
        ${COMMON_SOURCES}
        src/runner/ax_model_runner/ax_model_runner_axcl.cpp
        src/runner/utils/axcl_manager.cpp
    )
    target_include_directories(llm_smoke PRIVATE ${AXCL_INCLUDE_DIRS})
    target_compile_definitions(llm_smoke PRIVATE USE_AXCL)
    target_link_libraries(llm_smoke
        ${AXCL_LINK_LIBS}
        ${AXLLM_VISION_LIBS}
    )
    if(NOT WIN32)
        set_target_properties(axllm PROPERTIES LINK_FLAGS "-Wl,--as-needed -Wl,--unresolved-symbols=ignore-in-shared-libs")
        set_target_properties(llm_smoke PROPERTIES LINK_FLAGS "-Wl,--as-needed -Wl,--unresolved-symbols=ignore-in-shared-libs")
    endif()
    install(TARGETS axllm DESTINATION bin)
    install(TARGETS llm_smoke DESTINATION bin)
endif()

# ---------------------------------------------------------------------------
# Tokenizer smoke test (host runnable; no HW required)
# ---------------------------------------------------------------------------
add_executable(tokenizer_smoke tools/tokenizer_smoke.cpp)
target_link_libraries(tokenizer_smoke tokenizer)
install(TARGETS tokenizer_smoke DESTINATION bin)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/scripts/ DESTINATION bin/scripts)
