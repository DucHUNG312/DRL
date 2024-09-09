cmake_minimum_required(VERSION 3.25)

project(DRL VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS                 "-Wall"         CACHE STRING "C++ compiler flags")
    set(CMAKE_CXX_FLAGS_DEBUG           "-g"            CACHE STRING "C++ compiler debug flags")
    set(CMAKE_CXX_FLAGS_MINSIZEREL      "-Os -DNDEBUG"  CACHE STRING "C++ compiler min size release flags")
    set(CMAKE_CXX_FLAGS_RELEASE         "-O3 -DNDEBUG"  CACHE STRING "C++ compiler release flags")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO  "-O2 -g"        CACHE STRING "C++ compiler flags for release with debug info")
    set(CMAKE_AR                        "ar"            CACHE STRING "Archiver program")
    set(CMAKE_LINKER                    "ld"            CACHE STRING "Linker program")
    set(CMAKE_NM                        "nm"            CACHE STRING "Symbol table viewer")
    set(CMAKE_OBJDUMP                   "objdump"       CACHE STRING "Object dump program")
    set(CMAKE_RANLIB                    "ranlib"        CACHE STRING "Archive indexer")
endif()

if(PROJECT_IS_TOP_LEVEL)
  set(LAB_TOP_PROJECT ON)
else()
  set(LAB_TOP_PROJECT OFF)
endif()

option(LAB_BUILD_EXAMPLES   "Build examples"  ${LAB_TOP_PROJECT})
option(LAB_BUILD_TESTS      "Build tests"     ${LAB_TOP_PROJECT})
option(LAB_GPU              "GPU Support"     ON)

if (DEFINED CMAKE_MODULE_PATH)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} CACHE PATH "Path to the folder containing finders")
endif()

include(cmake/build_options.cmake)
include(cmake/paths.cmake)
include(cmake/utils.cmake)
if(LAB_GPU)
    include(cmake/select_compute_arch.cmake)
    include(cmake/cuda.cmake)
endif()

add_subdirectory("3rdparty")
include(cmake/dependencies.cmake)

add_subdirectory("lab")

if(LAB_BUILD_TESTS)
    add_subdirectory("test")
endif()

if(LAB_BUILD_EXAMPLES)
    add_subdirectory("example")
endif()


configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/labConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lab
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/labConfig.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lab
)