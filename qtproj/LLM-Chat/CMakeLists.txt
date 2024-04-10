cmake_minimum_required(VERSION 3.5)

project(LLM-Chat LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(Qt6 COMPONENTS Widgets)
if(Qt6_FOUND)
  set(QTWidget Qt6::Widgets)
else()
  find_package(Qt5 COMPONENTS Widgets REQUIRED)
  set(QTWidget Qt5::Widgets)
endif()

add_executable(LLM-Chat
  main.cpp
  mainwindow.cpp
  mainwindow.h
  mainwindow.ui
  style/DarkStyle.cpp
  src/runner/ax_model_runner/ax_model_runner_ax650.cpp 
  src/runner/utils/memory_utils.cpp 
  src/runner/utils/cqdm.cpp
  src/runner/Tokenizer/Tokenizer.cpp
  src/runner/Tokenizer/QwenTokenizer.cpp
)

target_link_libraries(LLM-Chat PRIVATE ${QTWidget})

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

target_link_libraries(LLM-Chat PRIVATE ax_engine ax_interpreter ax_sys ax_ivps)
target_link_libraries(LLM-Chat PRIVATE sentencepiece re2::re2)