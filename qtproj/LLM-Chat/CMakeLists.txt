cmake_minimum_required(VERSION 3.5)

project(LLM-Chat LANGUAGES CXX)

add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# bsp
if(NOT BSP_MSP_DIR)
    # 判断 /soc/lib/libax_engine.so 是否存在，以确定是否为板端编译
    if(EXISTS /soc/lib/libax_engine.so)
        message(STATUS "Detected board,BSP_MSP_DIR = /soc")
        set(BSP_MSP_DIR /soc)
    else()
        set(BSP_MSP_DIR ${CMAKE_SOURCE_DIR}/bsp_msp_out/msp/out)
    endif()
endif()

message(STATUS "BSP_MSP_DIR = ${BSP_MSP_DIR}")
include_directories(${BSP_MSP_DIR}/include)
link_directories(${BSP_MSP_DIR}/lib)

if(NOT OpenCV_DIR)
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/third_party/libopencv-4.6-aarch64-none/lib/cmake/opencv4)
endif()

message(STATUS "OpenCV_DIR Path: ${OpenCV_DIR}")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(axllm STATIC 
    src/runner/ax_model_runner/ax_model_runner_ax650.cpp 
    src/runner/utils/memory_utils.cpp 
    src/runner/utils/cqdm.cpp
    src/runner/Tokenizer/Tokenizer.cpp
    src/runner/Tokenizer/QwenTokenizer.cpp
)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 COMPONENTS Widgets)
if(Qt6_FOUND)
  message(STATUS "Found Qt6")
  set(QTWidget Qt6::Widgets)
else()
  find_package(Qt5 COMPONENTS Widgets REQUIRED)
  set(QTWidget Qt5::Widgets)
endif()

add_executable(LLM-Chat
  main.cpp
  mainwindow.cpp
  mainwindow.h
  myqlabel.h
  mainwindow.ui
  style/DarkStyle.cpp
)

target_link_libraries(LLM-Chat PRIVATE ${QTWidget})
target_link_libraries(LLM-Chat PRIVATE axllm)
target_link_libraries(LLM-Chat PRIVATE ax_engine ax_interpreter ax_sys ax_ivps)
target_link_libraries(LLM-Chat PRIVATE sentencepiece re2::re2)
target_link_libraries(LLM-Chat PRIVATE ${OpenCV_LIBS})