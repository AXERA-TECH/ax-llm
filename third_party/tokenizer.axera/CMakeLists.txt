if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()

message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

project(tokenizer.axera)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_library(tokenizer src/BaseTokenizer.cpp src/tokenizer/tokenizer_optimized.cpp)

function(add_test test_name)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} tokenizer)
endfunction()

# option to build tests
option(TOKENIZER_BUILD_TESTS "Build tests" ON)
MESSAGE(STATUS "TOKENIZER_BUILD_TESTS = ${TOKENIZER_BUILD_TESTS}")

if(TOKENIZER_BUILD_TESTS)
    add_test(test_tokenizer)
    add_test(test_num2words_en)
endif()


install(TARGETS tokenizer DESTINATION lib)
install(FILES include/BaseTokenizer.hpp DESTINATION include)