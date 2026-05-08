# set cross-compiled system type, it's better not use the type which cmake cannot recognized.
SET (CMAKE_SYSTEM_NAME Linux)
SET (CMAKE_SYSTEM_PROCESSOR riscv64)

set(RISCV64_GCC "" CACHE FILEPATH "Path to riscv64 gcc")
set(RISCV64_GXX "" CACHE FILEPATH "Path to riscv64 g++")

set(RISCV64_TOOLCHAIN_BIN "" CACHE PATH "Path to RISC-V toolchain bin directory")
if (NOT RISCV64_TOOLCHAIN_BIN)
    if (EXISTS "/home/axera/gcc-14.3-riscv64-unknown-linux-gnu-2.39/bin")
        set(RISCV64_TOOLCHAIN_BIN "/home/axera/gcc-14.3-riscv64-unknown-linux-gnu-2.39/bin")
    endif()
endif()

if (RISCV64_TOOLCHAIN_BIN AND EXISTS "${RISCV64_TOOLCHAIN_BIN}")
    list(PREPEND CMAKE_PROGRAM_PATH "${RISCV64_TOOLCHAIN_BIN}")
    if (NOT RISCV64_GCC)
        if (EXISTS "${RISCV64_TOOLCHAIN_BIN}/riscv64-unknown-linux-gnu-gcc")
            set(RISCV64_GCC "${RISCV64_TOOLCHAIN_BIN}/riscv64-unknown-linux-gnu-gcc")
        elseif (EXISTS "${RISCV64_TOOLCHAIN_BIN}/riscv64-linux-gnu-gcc")
            set(RISCV64_GCC "${RISCV64_TOOLCHAIN_BIN}/riscv64-linux-gnu-gcc")
        elseif (EXISTS "${RISCV64_TOOLCHAIN_BIN}/riscv64-none-linux-gnu-gcc")
            set(RISCV64_GCC "${RISCV64_TOOLCHAIN_BIN}/riscv64-none-linux-gnu-gcc")
        endif()
    endif()
    if (NOT RISCV64_GXX)
        if (EXISTS "${RISCV64_TOOLCHAIN_BIN}/riscv64-unknown-linux-gnu-g++")
            set(RISCV64_GXX "${RISCV64_TOOLCHAIN_BIN}/riscv64-unknown-linux-gnu-g++")
        elseif (EXISTS "${RISCV64_TOOLCHAIN_BIN}/riscv64-linux-gnu-g++")
            set(RISCV64_GXX "${RISCV64_TOOLCHAIN_BIN}/riscv64-linux-gnu-g++")
        elseif (EXISTS "${RISCV64_TOOLCHAIN_BIN}/riscv64-none-linux-gnu-g++")
            set(RISCV64_GXX "${RISCV64_TOOLCHAIN_BIN}/riscv64-none-linux-gnu-g++")
        endif()
    endif()
endif()

if (NOT RISCV64_GCC)
    # make sure riscv64-*-linux-gnu-gcc can be found in $PATH
    find_program(RISCV64_GCC
        NAMES riscv64-unknown-linux-gnu-gcc riscv64-linux-gnu-gcc riscv64-none-linux-gnu-gcc
    )
endif()
if (NOT RISCV64_GXX)
    find_program(RISCV64_GXX
        NAMES riscv64-unknown-linux-gnu-g++ riscv64-linux-gnu-g++ riscv64-none-linux-gnu-g++
    )
endif()

if (NOT RISCV64_GCC OR NOT RISCV64_GXX)
    message(FATAL_ERROR "RISC-V toolchain not found. Set RISCV64_GCC/RISCV64_GXX or ensure riscv64-*-linux-gnu-gcc/g++ is in PATH.")
endif()

SET (CMAKE_C_COMPILER   "${RISCV64_GCC}")
SET (CMAKE_CXX_COMPILER "${RISCV64_GXX}")

# Default AXCL SDK (RISC-V build) if present. Can be overridden by -DAXCL_DIR=...
if (NOT DEFINED AXCL_DIR)
    if (EXISTS "/home/axera/msp_dir/axcl_linux_riscv")
        set(AXCL_DIR "/home/axera/msp_dir/axcl_linux_riscv" CACHE PATH "AXCL SDK root (RISC-V)" )
        message(STATUS "AXCL_DIR (default) = ${AXCL_DIR}")
    endif()
endif()

# set searching rules for cross-compiler
SET (CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET (CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET (CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
