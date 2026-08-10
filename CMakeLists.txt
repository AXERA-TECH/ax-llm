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
# Build/version metadata
# ---------------------------------------------------------------------------
set(AXLLM_GIT_COMMIT "unknown")
set(AXLLM_GIT_BRANCH "unknown")
set(AXLLM_GIT_DIRTY "unknown")
set(AXLLM_TOKENIZER_GIT_COMMIT "unknown")
set(AXLLM_TOKENIZER_GIT_DIRTY "unknown")

find_package(Git QUIET)
if(Git_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short=12 HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE AXLLM_GIT_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE AXLLM_GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    execute_process(
        COMMAND ${GIT_EXECUTABLE} diff --quiet --ignore-submodules HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE AXLLM_GIT_DIRTY_RESULT
        ERROR_QUIET
    )
    if(AXLLM_GIT_DIRTY_RESULT EQUAL 0)
        set(AXLLM_GIT_DIRTY "no")
    else()
        set(AXLLM_GIT_DIRTY "yes")
    endif()

    if(EXISTS ${CMAKE_SOURCE_DIR}/third_party/tokenizer.axera/.git)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse --short=12 HEAD
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/tokenizer.axera
            OUTPUT_VARIABLE AXLLM_TOKENIZER_GIT_COMMIT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        execute_process(
            COMMAND ${GIT_EXECUTABLE} diff --quiet HEAD
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/tokenizer.axera
            RESULT_VARIABLE AXLLM_TOKENIZER_GIT_DIRTY_RESULT
            ERROR_QUIET
        )
        if(AXLLM_TOKENIZER_GIT_DIRTY_RESULT EQUAL 0)
            set(AXLLM_TOKENIZER_GIT_DIRTY "no")
        else()
            set(AXLLM_TOKENIZER_GIT_DIRTY "yes")
        endif()
    endif()
endif()

function(axllm_apply_build_info target backend)
    if(OpenMP_CXX_FOUND)
        set(AXLLM_OPENMP_STATUS "ON")
    else()
        set(AXLLM_OPENMP_STATUS "OFF")
    endif()

    target_compile_definitions(${target} PRIVATE
        AXLLM_GIT_COMMIT="${AXLLM_GIT_COMMIT}"
        AXLLM_GIT_BRANCH="${AXLLM_GIT_BRANCH}"
        AXLLM_GIT_DIRTY="${AXLLM_GIT_DIRTY}"
        AXLLM_TOKENIZER_GIT_COMMIT="${AXLLM_TOKENIZER_GIT_COMMIT}"
        AXLLM_TOKENIZER_GIT_DIRTY="${AXLLM_TOKENIZER_GIT_DIRTY}"
        AXLLM_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
        AXLLM_BUILD_BACKEND="${backend}"
        AXLLM_SYSTEM_NAME="${CMAKE_SYSTEM_NAME}"
        AXLLM_SYSTEM_PROCESSOR="${CMAKE_SYSTEM_PROCESSOR}"
        AXLLM_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
        AXLLM_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
        AXLLM_OPENMP="${AXLLM_OPENMP_STATUS}"
    )
endfunction()

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
    src/runner/LLM_embed.cpp
    src/runner/LLM_init.cpp
    src/runner/LLM_decode.cpp
    src/runner/image/sd15_image_generator.cpp
    src/runner/utils/files.cpp
    src/runner/utils/image_processor.cpp
    src/runner/utils/mrope.cpp
    src/runner/vision/vision_module.cpp
    src/runner/vision/vision_adapters.cpp
)

# ---------------------------------------------------------------------------
# AX650 片上后端  →  axllm
# ---------------------------------------------------------------------------
if(BUILD_AX650)
    add_executable(axllm src/main.cpp
        ${COMMON_SOURCES}
        src/runner/ax_model_runner/ax_model_runner_ax650.cpp
    )
    axllm_apply_build_info(axllm "AX650")
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
        pthread ${CMAKE_DL_LIBS} tokenizer openai_api::server
        ${AXLLM_VISION_LIBS}
    )
    target_link_libraries(llm_smoke
        -L${BSP_MSP_DIR}/lib
        ax_engine ax_interpreter ax_sys
        pthread ${CMAKE_DL_LIBS} tokenizer openai_api::server
        ${AXLLM_VISION_LIBS}
    )
    add_executable(golden_runner tools/golden_runner.cpp
        ${COMMON_SOURCES}
        src/runner/ax_model_runner/ax_model_runner_ax650.cpp
    )
    target_include_directories(golden_runner PRIVATE ${BSP_MSP_DIR}/include)
    target_link_libraries(golden_runner
        -L${BSP_MSP_DIR}/lib
        ax_engine ax_interpreter ax_sys
        pthread ${CMAKE_DL_LIBS} tokenizer openai_api::server
        ${AXLLM_VISION_LIBS}
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(axllm OpenMP::OpenMP_CXX)
        target_link_libraries(llm_smoke OpenMP::OpenMP_CXX)
        target_link_libraries(golden_runner OpenMP::OpenMP_CXX)
    endif()
    install(TARGETS axllm DESTINATION bin)
    install(TARGETS llm_smoke DESTINATION bin)
    install(TARGETS golden_runner DESTINATION bin)
endif()

# ---------------------------------------------------------------------------
# AXCL PCIe 后端  →  axllm
# ---------------------------------------------------------------------------
if(BUILD_AXCL)
    set(AXCL_LINK_LIBS tokenizer openai_api::server Threads::Threads ${CMAKE_DL_LIBS})

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
    axllm_apply_build_info(axllm "AXCL")
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
    add_executable(golden_runner tools/golden_runner.cpp
        ${COMMON_SOURCES}
        src/runner/ax_model_runner/ax_model_runner_axcl.cpp
        src/runner/utils/axcl_manager.cpp
    )
    target_include_directories(golden_runner PRIVATE ${AXCL_INCLUDE_DIRS})
    target_compile_definitions(golden_runner PRIVATE USE_AXCL)
    target_link_libraries(golden_runner
        ${AXCL_LINK_LIBS}
        ${AXLLM_VISION_LIBS}
    )
    if(NOT WIN32)
        set_target_properties(axllm PROPERTIES LINK_FLAGS "-Wl,--as-needed -Wl,--unresolved-symbols=ignore-in-shared-libs")
        set_target_properties(llm_smoke PROPERTIES LINK_FLAGS "-Wl,--as-needed -Wl,--unresolved-symbols=ignore-in-shared-libs")
        set_target_properties(golden_runner PROPERTIES LINK_FLAGS "-Wl,--as-needed -Wl,--unresolved-symbols=ignore-in-shared-libs")
    endif()
    install(TARGETS axllm DESTINATION bin)
    install(TARGETS llm_smoke DESTINATION bin)
    install(TARGETS golden_runner DESTINATION bin)
endif()

# ---------------------------------------------------------------------------
# Tokenizer smoke test (host runnable; no HW required)
# ---------------------------------------------------------------------------
add_executable(tokenizer_smoke tools/tokenizer_smoke.cpp)
target_link_libraries(tokenizer_smoke tokenizer)
install(TARGETS tokenizer_smoke DESTINATION bin)

# Multi-slot prefix-KV selection policy unit test (host-runnable; no HW required).
# Pins the 4-strategy + LRU decision used by select_kv_slot / select_kv_slot_by_history.
add_executable(kv_slot_select_test tools/kv_slot_select_test.cpp)
target_include_directories(kv_slot_select_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(kv_slot_select_test tokenizer)
install(TARGETS kv_slot_select_test DESTINATION bin)

# Vision preprocessing layout regression test (host-runnable; no HW required).
# Guards the Qwen-VL patchifier (Qwen2VideoProcessor) against channel-planar
# layout regressions that scramble the vision encoder input.
add_executable(vision_preprocess_test tools/vision_preprocess_test.cpp
    src/runner/utils/image_processor.cpp
    src/runner/utils/files.cpp)
target_include_directories(vision_preprocess_test PRIVATE src/runner/utils)
target_link_libraries(vision_preprocess_test ${AXLLM_VISION_LIBS})
install(TARGETS vision_preprocess_test DESTINATION bin)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/scripts/ DESTINATION bin/scripts)
