###############################################################
#
# OverLook: Amplify C/C++ warnings that shouldn't be ignored.
#
# Author:   Zhuo Zhang <imzhuo@foxmail.com>
# Homepage: https://github.com/zchrissirhcz/overlook
#
###############################################################

cmake_minimum_required(VERSION 3.1)

# Only included once
if(OVERLOOK_INCLUDE_GUARD)
  return()
endif()
set(OVERLOOK_INCLUDE_GUARD TRUE)

set(OVERLOOK_VERSION "2023.06.13")

option(OVERLOOK_FLAGS_GLOBAL "use safe compilation flags?" ON)
option(OVERLOOK_USE_STRICT_FLAGS "strict c/c++ flags checking?" OFF)
option(OVERLOOK_USE_CPPCHECK "use cppcheck for static checking?" OFF)
option(OVERLOOK_VERBOSE "verbose output?" ON)

set(OVERLOOK_C_FLAGS "")
set(OVERLOOK_CXX_FLAGS "")

# Append element to list with space as seperator
function(overlook_list_append __string __element)
  # set(__list ${${__string}})
  # set(__list "${__list} ${__element}")
  # set(${__string} ${__list} PARENT_SCOPE)
  #set(__list ${${__string}})
  set(${__string} "${${__string}} ${__element}" PARENT_SCOPE)
endfunction()

# Print overlook information
message(STATUS "------------------------------------------------------------")
message(STATUS "  Using overlook, the CMake plugin for safer C/C++ code")
message(STATUS "  Author: Zhuo Zhang (imzhuo@foxmail.com)")
message(STATUS "  OVERLOOK_VERSION: ${OVERLOOK_VERSION}")
message(STATUS "------------------------------------------------------------")

# If `-w` specified for GCC/Clang, report an error
if((CMAKE_C_COMPILER_ID MATCHES "GNU") OR (CMAKE_C_COMPILER_ID MATCHES "Clang"))
  get_directory_property(overlook_detected_global_compile_options COMPILE_OPTIONS)
  message(STATUS "Overlook Detected Global Compile Options: ${overlook_detected_global_compile_options}")
  string(REGEX MATCH "-w" ignore_all_warnings "${overlook_detected_global_compile_options}" )
  if(ignore_all_warnings)
    message(FATAL_ERROR "Overlook won't working due to `-w` found in compile options. Consider remove it (in `add_compile_options)`")
  endif()
endif()

# 检查编译器版本。同一编译器的不同版本，编译选项的支持情况可能不同。
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CLANG_VERSION_STRING)
  message(STATUS "--- CLANG_VERSION_MAJOR is: ${CLANG_VERSION_MAJOR}")
  message(STATUS "--- CLANG_VERSION_MINOR is: ${CLANG_VERSION_MINOR}")
  message(STATUS "--- CLANG_VERSION_PATCHLEVEL is: ${CLANG_VERSION_PATCHLEVEL}")
  message(STATUS "--- CLANG_VERSION_STRING is: ${CLANG_VERSION_STRING}")
endif()

if(CMAKE_C_COMPILER_ID)
  set(OVERLOOK_WITH_C TRUE)
else()
  set(OVERLOOK_WITH_C FALSE)
endif()

if(CMAKE_CXX_COMPILER_ID)
  set(OVERLOOK_WITH_CXX TRUE)
else()
  set(OVERLOOK_WITH_CXX FALSE)
endif()

# Project LANGUAGE not including C and CXX so we return
if((NOT OVERLOOK_WITH_C) AND (NOT OVERLOOK_WITH_CXX))
  message("OverLook WARNING: neither C nor CXX compilers available. No OVERLOOK C/C++ flags will be set")
  message("  NOTE: You many consider add C and CXX in `project()` command")
  return()
endif()

option(OVERLOOK_ENABLE_RULE1  "enable rule1?"  ON)
option(OVERLOOK_ENABLE_RULE2  "enable rule2?"  ON)
option(OVERLOOK_ENABLE_RULE3  "enable rule3?"  OFF)
option(OVERLOOK_ENABLE_RULE4  "enable rule4?"  ON)
option(OVERLOOK_ENABLE_RULE5  "enable rule5?"  OFF)
option(OVERLOOK_ENABLE_RULE6  "enable rule6?"  ON)
option(OVERLOOK_ENABLE_RULE7  "enable rule7?"  ON)
option(OVERLOOK_ENABLE_RULE8  "enable rule8?"  OFF)

option(OVERLOOK_ENABLE_RULE10 "enable rule10?" OFF)
option(OVERLOOK_ENABLE_RULE11 "enable rule11?" ON)
option(OVERLOOK_ENABLE_RULE12 "enable rule12?" OFF)
option(OVERLOOK_ENABLE_RULE13 "enable rule13?" OFF)
option(OVERLOOK_ENABLE_RULE14 "enable rule14?" OFF)
option(OVERLOOK_ENABLE_RULE15 "enable rule15?" OFF)
option(OVERLOOK_ENABLE_RULE16 "enable rule16?" ON)
option(OVERLOOK_ENABLE_RULE17 "enable rule17?" OFF)
option(OVERLOOK_ENABLE_RULE18 "enable rule18?" OFF)
option(OVERLOOK_ENABLE_RULE19 "enable rule19?" OFF)

option(OVERLOOK_ENABLE_RULE21 "enable rule21?" OFF)
option(OVERLOOK_ENABLE_RULE22 "enable rule22?" OFF)
option(OVERLOOK_ENABLE_RULE23 "enable rule23?" OFF)
option(OVERLOOK_ENABLE_RULE24 "enable rule24?" OFF)
option(OVERLOOK_ENABLE_RULE25 "enable rule25?" OFF)
option(OVERLOOK_ENABLE_RULE26 "enable rule26?" OFF)
option(OVERLOOK_ENABLE_RULE27 "enable rule27?" OFF)
option(OVERLOOK_ENABLE_RULE28 "enable rule28?" OFF)
option(OVERLOOK_ENABLE_RULE29 "enable rule29?" OFF)
option(OVERLOOK_ENABLE_RULE30 "enable rule30?" OFF)
option(OVERLOOK_ENABLE_RULE31 "enable rule31?" OFF)
option(OVERLOOK_ENABLE_RULE32 "enable rule32?" OFF)
option(OVERLOOK_ENABLE_RULE33 "enable rule33?" OFF)

option(OVERLOOK_ENABLE_RULE35 "enable rule35?" OFF)
option(OVERLOOK_ENABLE_RULE36 "enable rule36?" OFF)

if(OVERLOOK_USE_STRICT_FLAGS)
  option(OVERLOOK_ENABLE_RULE9  "enable rule9?"   ON)
  option(OVERLOOK_ENABLE_RULE20 "enable rule20?"  ON)
  option(OVERLOOK_ENABLE_RULE34 "enable rule34?"  ON)
else()
  option(OVERLOOK_ENABLE_RULE9  "enable rule9?"   OFF)
  option(OVERLOOK_ENABLE_RULE20 "enable rule20?"  OFF)
  option(OVERLOOK_ENABLE_RULE34 "enable rule34?"  OFF)
endif()

# Rule 1. 函数没有声明就使用
# 解决bug：地址截断；内存泄漏
if(OVERLOOK_ENABLE_RULE1)
  if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    overlook_list_append(OVERLOOK_C_FLAGS /we4013)
    overlook_list_append(OVERLOOK_CXX_FLAGS /we4013)
  elseif(CMAKE_C_COMPILER_ID MATCHES "GNU")
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=implicit-function-declaration)
    if(CMAKE_CXX_COMPILER_VERSION LESS 9.1)
      # overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=implicit-function-declaration)
    endif()
  elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=implicit-function-declaration)
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=implicit-function-declaration)
  endif()
endif()

# Rule 2. 函数虽然有声明，但是声明不完整，没有写出返回值类型
if(OVERLOOK_ENABLE_RULE2)
  if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    overlook_list_append(OVERLOOK_C_FLAGS /we4431)
    overlook_list_append(OVERLOOK_CXX_FLAGS /we4431)
  elseif(CMAKE_C_COMPILER_ID MATCHES "GNU")
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=implicit-int)
    if(CMAKE_CXX_COMPILER_VERSION LESS 9.1)
      # overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=implicit-int)
    endif()
  elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=implicit-int)
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=implicit-int)
  endif()
endif()

# Rule 3. 指针类型不兼容
# 解决bug：crash或结果异常
if(OVERLOOK_ENABLE_RULE3)
  if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    overlook_list_append(OVERLOOK_C_FLAGS /we4133)
    overlook_list_append(OVERLOOK_CXX_FLAGS /we4133)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    if(CMAKE_CXX_COMPILER_VERSION GREATER 4.8) # gcc/g++ 4.8.3 not ok
      overlook_list_append(OVERLOOK_C_FLAGS -Werror=incompatible-pointer-types)
      if(CMAKE_CXX_COMPILER_VERSION LESS 9.1)
        overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=incompatible-pointer-types)
      endif()
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=incompatible-pointer-types)
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=incompatible-pointer-types)
  endif()
endif()

# Rule 4. 函数应该有返回值但没有 return 返回值;或不是所有路径都有返回值
# 解决bug：lane detect; vpdt for循环无法跳出(android输出trap); lane calib库读取到随机值导致获取非法格式asvl，开asan则表现为读取NULL指针
# -O3时输出内容和其他优化等级不一样（from 三老师）
if(OVERLOOK_ENABLE_RULE4)
  if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    overlook_list_append(OVERLOOK_C_FLAGS /we4716 /we4715)
    overlook_list_append(OVERLOOK_CXX_FLAGS /we4716 /we4715)
  else()
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=return-type)
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=return-type)
  endif()
endif()

# Rule 5. 避免使用影子(shadow)变量
# 有时候会误伤，例如eigen等开源项目，可以手动关掉
if(OVERLOOK_ENABLE_RULE5)
  if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    overlook_list_append(OVERLOOK_C_FLAGS /we6244 /we6246 /we4457 /we4456)
    overlook_list_append(OVERLOOK_CXX_FLAGS /we6244 /we6246 /we4457 /we4456)
  else()
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=shadow)
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=shadow)
  endif()
endif()

# Rule 6. 函数不应该返回局部变量的地址
if(OVERLOOK_ENABLE_RULE6)
  if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    overlook_list_append(OVERLOOK_C_FLAGS /we4172)
    overlook_list_append(OVERLOOK_CXX_FLAGS /we4172)
  elseif(CMAKE_C_COMPILER_ID MATCHES "GNU")
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=return-local-addr)
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=return-local-addr)
  elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=return-stack-address)
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=return-stack-address)
  endif()
endif()

# Rule 7. 变量没初始化就使用，要避免
if(OVERLOOK_ENABLE_RULE7)
  if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    overlook_list_append(OVERLOOK_C_FLAGS "/we4700 /we26495")
    overlook_list_append(OVERLOOK_CXX_FLAGS "/we4700 /we26495")
  else()
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=uninitialized)
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=uninitialized)
  endif()
endif()

# Rule 8. printf 等语句中的格式串和实参类型不匹配，要避免
if(OVERLOOK_ENABLE_RULE8)
  if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    overlook_list_append(OVERLOOK_C_FLAGS /we4477)
    overlook_list_append(OVERLOOK_CXX_FLAGS /we4477)
  else()
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=format)
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=format)
  endif()
endif()

# Rule 9. 避免把 unsigned int 和 int 直接比较
# 通常会误伤，例如 for 循环中。可以考虑关掉
if(OVERLOOK_ENABLE_RULE9)
  if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    overlook_list_append(OVERLOOK_C_FLAGS /we4018)
    overlook_list_append(OVERLOOK_CXX_FLAGS /we4018)
  elseif(CMAKE_C_COMPILER_ID MATCHES "GNU")
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=sign-compare)
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=sign-compare)
  elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=sign-compare)
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=sign-compare)
  endif()
endif()

# Rule 10. 避免把 int 指针赋值给 int 类型变量
if(OVERLOOK_ENABLE_RULE10)
  if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    overlook_list_append(OVERLOOK_C_FLAGS /we4047)
    overlook_list_append(OVERLOOK_CXX_FLAGS /we4047)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    if(CMAKE_CXX_COMPILER_VERSION GREATER 4.8)
      overlook_list_append(OVERLOOK_C_FLAGS -Werror=int-conversion)
      if(CMAKE_CXX_COMPILER_VERSION LESS 9.1)
        overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=int-conversion)
      endif()
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=int-conversion)
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=int-conversion)
  endif()
endif()

# Rule 11. 检查数组下标越界访问
if(OVERLOOK_ENABLE_RULE11)
  if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    overlook_list_append(OVERLOOK_C_FLAGS "/we6201 /we6386 /we4789")
    overlook_list_append(OVERLOOK_CXX_FLAGS "/we6201 /we6386 /we4789")
  else()
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=array-bounds)
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=array-bounds)
  endif()
endif()

# Rule 12. 函数声明中的参数列表和定义中不一样。在 MSVC C 下为警告，Linux Clang 下报错
if(OVERLOOK_ENABLE_RULE12)
  if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    overlook_list_append(OVERLOOK_C_FLAGS /we4029)
  endif()
endif()

# Rule 13. 实参太多，比函数定义或声明中的要多。只在MSVC C 下为警告，Linux Clang下报错
if(OVERLOOK_ENABLE_RULE13)
  if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    overlook_list_append(OVERLOOK_C_FLAGS /we4020)
  endif()
endif()

# 14. 避免 void* 类型的指针参参与算术运算
# MSVC C/C++ 默认会报错，Linux gcc 不报 warning 和 error，Linux g++ 只报 warning
# Linux 下 Clang 开 -Wpedentric 才报 warning，Clang++ 报 error
if(OVERLOOK_ENABLE_RULE14)
  if(CMAKE_C_COMPILER_ID MATCHES "GNU")
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=pointer-arith)
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=pointer-arith)
  elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=pointer-arith)
  endif()
endif()

# Rule 15. 避免符号重复定义（变量对应的强弱符号）。只在 C 中出现。
# 暂时没找到 MSVC 的对应编译选项
if(OVERLOOK_ENABLE_RULE15)
  if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
    overlook_list_append(OVERLOOK_C_FLAGS -fno-common)
  endif()
endif()

# Rule 16. 释放非堆内存
# TODO: 检查 MSVC
# Linux Clang8.0 无法检测到
if(OVERLOOK_ENABLE_RULE16)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=free-nonheap-object)
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=free-nonheap-object)
  endif()
endif()

# Rule 17. 形参与声明不同。场景：静态库(.h/.c)，集成时换库但没换头文件，且函数形参有变化（类型或数量）
# 只报 warning 不报 error。仅 VS 出现
if(OVERLOOK_ENABLE_RULE17)
  if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    overlook_list_append(OVERLOOK_C_FLAGS /we4028)
  endif()
endif()

# Rule 18. 宏定义重复
# gcc5~gcc9 无法检查
if(OVERLOOK_ENABLE_RULE18)
  if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    overlook_list_append(OVERLOOK_C_FLAGS /we4005)
    overlook_list_append(OVERLOOK_CXX_FLAGS /we4005)
  elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=macro-redefined)
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=macro-redefined)
  endif()
endif()

# Rule 19. pragma init_seg 指定了非法（不能识别的）section名字
# VC++ 特有。Linux 下的 gcc/clang 没有
if(OVERLOOK_ENABLE_RULE19)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    overlook_list_append(OVERLOOK_CXX_FLAGS /we4075)
  endif()
endif()

# Rule 20. size_t 类型被转为更窄类型
# VC/VC++ 特有。 Linux 下的 gcc/clang 没有
# 有点过于严格了
if(OVERLOOK_ENABLE_RULE20)
  if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    overlook_list_append(OVERLOOK_C_FLAGS /we4267)
    overlook_list_append(OVERLOOK_CXX_FLAGS /we4267)
  endif()
endif()

# Rule 21. “类型强制转换”: 例如从 int 转换到更大的 void *
if(OVERLOOK_ENABLE_RULE21)
  if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    overlook_list_append(OVERLOOK_C_FLAGS /we4312)
    overlook_list_append(OVERLOOK_CXX_FLAGS /we4312)
  elseif(CMAKE_C_COMPILER_ID MATCHES "GNU")
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=int-to-pointer-cast)
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=int-to-pointer-cast)
  elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=int-to-pointer-cast)
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=int-to-pointer-cast)
  endif()
endif()

# Rule 22. 不可识别的字符转义序列
# GCC5.4 能显示 warning 但无别名，因而无法视为 error
if(OVERLOOK_ENABLE_RULE22)
  if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    overlook_list_append(OVERLOOK_C_FLAGS /we4129)
    overlook_list_append(OVERLOOK_CXX_FLAGS /we4129)
  elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=unknown-escape-sequence)
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=unknown-escape-sequence)
  endif()
endif()

# Rule 23. 类函数宏的调用 参数过多
# VC/VC++ 报警告。Linux 下的 GCC/Clang 报 error
if(OVERLOOK_ENABLE_RULE23)
  if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    overlook_list_append(OVERLOOK_C_FLAGS /we4002)
    overlook_list_append(OVERLOOK_CXX_FLAGS /we4002)
  endif()
endif()

# Rule 24. 类函数宏的调用 参数不足
# VC/VC++ 同时会报 error C2059
# Linux GCC/Clang 直接报错
if(OVERLOOK_ENABLE_RULE24)
  if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    overlook_list_append(OVERLOOK_C_FLAGS /we4003)
    overlook_list_append(OVERLOOK_CXX_FLAGS /we4003)
  endif()
endif()

# Rule 25. #undef 没有跟一个标识符
# Linux GCC/Clang 直接报错
if(OVERLOOK_ENABLE_RULE25)
  if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    overlook_list_append(OVERLOOK_C_FLAGS /we4006)
    overlook_list_append(OVERLOOK_CXX_FLAGS /we4006)
  endif()
endif()

# Rule 26. 单行注释包含行继续符
# 可能会导致下一行代码报错，而问题根源在包含继续符的这行注释
if(OVERLOOK_ENABLE_RULE26)
  if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    overlook_list_append(OVERLOOK_C_FLAGS /we4006)
    overlook_list_append(OVERLOOK_CXX_FLAGS /we4006)
  elseif(CMAKE_C_COMPILER_ID MATCHES "GNU")
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=comment)
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=comment)
  elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=comment)
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=comment)
  endif()
endif()

# Rule 27. 没有使用到表达式结果（无用代码行，应删除）
# 感觉容易被误伤，可以考虑关掉
if(OVERLOOK_ENABLE_RULE27)
  if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    overlook_list_append(OVERLOOK_C_FLAGS "/we4552 /we4555")
    overlook_list_append(OVERLOOK_CXX_FLAGS "/we4552 /we4555")
  elseif(CMAKE_C_COMPILER_ID MATCHES "GNU")
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=unused-value)
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=unused-value)
  elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=unused-value)
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=unused-value)
  endif()
endif()

# Rule 28. “==”: 未使用表达式结果；是否打算使用“=”?
# Linux GCC 没有对应的编译选项
if(OVERLOOK_ENABLE_RULE28)
  if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    overlook_list_append(OVERLOOK_C_FLAGS /we4553)
    overlook_list_append(OVERLOOK_CXX_FLAGS /we4553)
  elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=unused-comparison)
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=unused-comparison)
  endif()
endif()

# Rule 29. C++中，禁止把字符串常量赋值给 char* 变量
# VS2019 开启 /Wall 后也查不到
if(OVERLOOK_ENABLE_RULE29)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=write-strings)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Linux Clang 和 AppleClang 不太一样
    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
      overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=writable-strings)
    elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
      overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=c++11-compat-deprecated-writable-strings)
    endif()
  endif()
endif()

# Rule 30. 所有的控件路径(if/else)必须都有返回值
# NDK21 Clang / Linux Clang/GCC/G++ 默认都报 error
if(OVERLOOK_ENABLE_RULE30)
  if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    overlook_list_append(OVERLOOK_C_FLAGS /we4715)
    overlook_list_append(OVERLOOK_CXX_FLAGS /we4715)
  endif()
endif()

# Rule 31. multi-char constant
# MSVC 没有对应的选项
if(OVERLOOK_ENABLE_RULE31)
  if(CMAKE_C_COMPILER_ID MATCHES "GNU")
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=multichar)
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=multichar)
  elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=multichar)
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=multichar)
  endif()
endif()

# Rule 32. 用 memset 等 C 函数设置 非 POD class 对象
# Linux下，GCC9.3 能发现此问题，但clang10 不能发现
if(OVERLOOK_ENABLE_RULE32)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    if(CMAKE_CXX_COMPILER_VERSION GREATER 7.5)
      overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=class-memaccess)
    endif()
  endif()
endif()

## Rule 33. 括号里面是单个等号而不是双等号
# Linux下， Clang14 可以发现问题，但 GCC9.3 无法发现；android clang 可以发现
if(OVERLOOK_ENABLE_RULE33)
  if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    overlook_list_append(OVERLOOK_C_FLAGS -Werror=parentheses)
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=parentheses)
  endif()
endif()

## Rule 34. double 型转 float 型，可能有精度丢失（尤其在 float 较大时）
# MSVC 默认是放在 /W3
if(OVERLOOK_ENABLE_RULE34)
  if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    overlook_list_append(OVERLOOK_C_FLAGS /we4244)
    overlook_list_append(OVERLOOK_CXX_FLAGS /we4244)
  endif()
endif()

## Rule 35. 父类有 virtual 的成员函数，但析构函数是 public 并且不是 virtual，会导致 UB
# https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#c35-a-base-class-destructor-should-be-either-public-and-virtual-or-protected-and-non-virtual
# -Wnon-virtual-dtor (C++ and Objective-C++ only)
# Warn when a class has virtual functions and an accessible non-virtual destructor itself or in an accessible polymorphic base
# class, in which case it is possible but unsafe to delete an instance of a derived class through a pointer to the class
# itself or base class.  This warning is automatically enabled if -Weffc++ is specified.
if(OVERLOOK_ENABLE_RULE35)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=non-virtual-dtor)
  elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    overlook_list_append(OVERLOOK_CXX_FLAGS -Werror=non-virtual-dtor)
  endif()
endif()

# 将上述定制的 FLAGS 追加到 CMAKE 默认的编译选项中
# 为什么是添加而不是直接设定呢？因为 xxx-toolchain.cmake 中可能会设置一些默认值 (如 Android NDK)， 需要避免这些默认值被覆盖
if(OVERLOOK_FLAGS_GLOBAL)
  overlook_list_append(CMAKE_C_FLAGS "${OVERLOOK_C_FLAGS}")
  overlook_list_append(CMAKE_CXX_FLAGS "${OVERLOOK_CXX_FLAGS}")
endif()

if(OVERLOOK_VERBOSE)
  message(STATUS "--- OVERLOOK_C_FLAGS are: ${OVERLOOK_C_FLAGS}")
  message(STATUS "--- OVERLOOK_CXX_FLAGS are: ${OVERLOOK_CXX_FLAGS}")
endif()


##################################################################################
# Add whole archive when build static library
# Usage:
#   overlook_add_whole_archive_flag(<lib> <output_var>)
# Example:
#   add_library(foo foo.hpp foo.cpp)
#   add_executable(bar bar.cpp)
#   overlook_add_whole_archive_flag(foo safe_foo)
#   target_link_libraries(bar ${safe_foo})
##################################################################################
function(overlook_add_whole_archive_flag lib output_var)
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
    if(MSVC_VERSION GREATER 1900)
      set(${output_var} -WHOLEARCHIVE:$<TARGET_FILE:${lib}> PARENT_SCOPE)
    else()
      message(WARNING "MSVC version is ${MSVC_VERSION}, /WHOLEARCHIVE flag cannot be set")
      set(${output_var} ${lib} PARENT_SCOPE)
    endif()
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    set(${output_var} -Wl,--whole-archive ${lib} -Wl,--no-whole-archive PARENT_SCOPE)
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(${output_var} -Wl,--whole-archive ${lib} -Wl,--no-whole-archive PARENT_SCOPE)
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND NOT ANDROID)
    set(${output_var} -Wl,-force_load ${lib} PARENT_SCOPE)
  elseif(ANDROID)
    #即使是 NDK21 并且手动传入 ANDROID_LD=lld，依然要用ld的查重复符号的链接选项
    set(${output_var} -Wl,--whole-archive ${lib} -Wl,--no-whole-archive PARENT_SCOPE)
  else()
    message(FATAL_ERROR ">>> add_whole_archive_flag not supported yet for current compiler: ${CMAKE_CXX_COMPILER_ID}")
  endif()
endfunction()


###############################################################
#
# cppcheck，开启静态代码检查，主要是检查编译器检测不到的UB
#   注: 目前只有终端下能看到对应输出，其中 NDK 下仅第一次输出
#
###############################################################
if(OVERLOOK_USE_CPPCHECK)
  find_program(CMAKE_CXX_CPPCHECK NAMES cppcheck)
  if(CMAKE_CXX_CPPCHECK)
    message(STATUS "cppcheck found")
    list(APPEND CMAKE_CXX_CPPCHECK
      "--enable=warning"
      "--inconclusive"
      "--force"
      "--inline-suppr"
    )
  else()
    message(STATUS "cppcheck not found. ignore it")
  endif()
endif()