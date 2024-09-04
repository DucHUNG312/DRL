cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

# Check CCACHE
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "Enabled CCache.")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

project(RL VERSION 1.0.0 LANGUAGES C CXX CUDA)

# Check LTO
include(CheckIPOSupported)
check_ipo_supported(
    RESULT LAB_IPO_SUPPORT 
    OUTPUT LAB_IPO_ERROR
)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

option(LAB_GPU "GPU Support" ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "DEBUG" CACHE STRING "Choose the type of build." FORCE)
endif()
if(NOT LAB_GPU)
    set(LAB_DEVICE "CPU" CACHE STRING "Choose the CPU." FORCE)
else()
    set(LAB_DEVICE "GPU" CACHE STRING "Choose the GPU." FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(${CMAKE_BUILD_TYPE} STREQUAL "DEBUG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -DNDEBUG")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    add_definitions(-DLAB_DEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    add_definitions(-DLAB_RELEASE)
endif()

find_package(CUDAToolkit)
if(LAB_DEVICE STREQUAL "GPU")
    if (NOT CUDAToolkit_FOUND)
        message(WARNING "GPU build is set but CUDA toolkit not found. Use CPU instead.")
        set(LAB_DEVICE "CPU")
    else()
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURE)
            set(CMAKE_CUDA_ARCHITECTURE "52")
            message(WARNING "CMAKE_CUDA_ARCHITECTURE is not set. Use default architecture.")
        endif()
        message(STATUS "CMAKE_CUDA_ARCHITECTURE: ${CMAKE_CUDA_ARCHITECTURE}")
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        # Add CUDA flags
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fpermissive --expt-relaxed-constexpr --expt-extended-lambda -allow-unsupported-compiler -gencode arch=compute_${CMAKE_CUDA_ARCHITECTURE},code=sm_${CMAKE_CUDA_ARCHITECTURE}")
        add_definitions(-DLAB_GPU_BUILD)
    endif()
endif()

find_package(OpenMP REQUIRED)
# Set OpenMP flags for a few compilers
if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang") # using Clang
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive -fstrict-aliasing -march=native -mtune=native")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Rpass-analysis=loop-vectorize")
   set(CMAKE_NOVEC_FLAGS "${CMAKE_NOVEC_FLAGS} -fnovectorize")
   set(CMAKE_VEC_FLAGS "${CMAKE_VEC_FLAGS} -fvectorize")
   set(CMAKE_OPENMP_FLAGS "${CMAKE_OPENMP_FLAGS} -fopenmp")
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU") # using GCC
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive -fstrict-aliasing -march=native -mtune=native")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopt-info-vec-optimized -fopt-info-omp-optimized")
   set(CMAKE_NOVEC_FLAGS "${CMAKE_NOVEC_FLAGS} -fno-tree-vectorize")
   set(CMAKE_VEC_FLAGS "${CMAKE_VEC_FLAGS} -ftree-vectorize")
   set(CMAKE_OPENMP_FLAGS "${CMAKE_OPENMP_FLAGS} -fopenmp")
   if ("${CMAKE_C_COMPILER_VERSION}" VERSION_GREATER "7.4.0")
      set(CMAKE_VEC_FLAGS "${CMAKE_VEC_FLAGS} -mprefer-vector-width=512")
   endif ("${CMAKE_C_COMPILER_VERSION}" VERSION_GREATER "7.4.0")
elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel") # using Intel C
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ansi-alias")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-report-phase=openmp,loop,vec")
   set(CMAKE_NOVEC_FLAGS "${CMAKE_NOVEC_FLAGS} -no-vec")
   set(CMAKE_VEC_FLAGS "${CMAKE_VEC_FLAGS} -xHOST")
   if ("${CMAKE_C_COMPILER_VERSION}" VERSION_GREATER "17.0.4")
     set(CMAKE_VEC_FLAGS "${CMAKE_VEC_FLAGS} -qopt-zmm-usage=high")
   endif ("${CMAKE_C_COMPILER_VERSION}" VERSION_GREATER "17.0.4")
elseif (CMAKE_C_COMPILER_ID MATCHES "MSVC")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Qvec-report:2")
   set(CMAKE_NOVEC_FLAGS "${CMAKE_NOVEC_FLAGS}")
   set(CMAKE_VEC_FLAGS "${CMAKE_VEC_FLAGS}")
elseif (CMAKE_C_COMPILER_ID MATCHES "XL")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qalias=ansi -qhot -qarch=pwr9 -qtune=pwr9")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qreport")
   set(CMAKE_NOVEC_FLAGS "${CMAKE_NOVEC_FLAGS} -qsimd=noauto")
   set(CMAKE_VEC_FLAGS "${CMAKE_VEC_FLAGS} -qsimd=auto")
elseif (CMAKE_C_COMPILER_ID MATCHES "Cray")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -h restrict=a")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -h msgs -h negmsgs -h list=a")
   set(CMAKE_NOVEC_FLAGS "${CMAKE_NOVEC_FLAGS} -h vector=0")
   set(CMAKE_VEC_FLAGS "${CMAKE_VEC_FLAGS} -h vector=3")
endif()

# Pytorch
set(PYTORCH_VERSION "2.4.0")
set(PYTORCH_MIN_VERSION "1.12.0")
find_package(Torch QUIET PATHS "${CMAKE_SOURCE_DIR}/libtorch")
if((NOT Torch_FOUND) OR (("${Torch_VERSION}" VERSION_LESS "${PYTORCH_MIN_VERSION}") OR
                         ("${Torch_VERSION}" VERSION_GREATER "${PYTORCH_VERSION}")))
    unset(Torch_FOUND)
    message(STATUS "Could not find compatible Torch version (>= ${PYTORCH_MIN_VERSION}, <= ${PYTORCH_VERSION})")
    include(fetch_libtorch)
endif()

message(STATUS "Torch version ${Torch_VERSION}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory("extern")
find_package(spdlog REQUIRED)
find_package(OpenGL)
find_package(LZ4 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Libuuid REQUIRED)

add_compile_definitions(EXPERIMENT_SPEC_DIR="${CMAKE_SOURCE_DIR}/spec/experiment/")

if(LAB_IPO_SUPPORT)
    message(STATUS "Enabled LTO.")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(STATUS "LTO is not supported: ${LAB_IPO_ERROR}.")
    message(STATUS "Compile without LTO.")
endif()

# common
set(COMMON_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/common/include")
set(COMMON_LINKED_LIBRARIES 
    spdlog::spdlog_header_only)
add_subdirectory("common")

# renderer
set(RENDERER_INCLUDE_DIRS
    ${COMMON_INCLUDE_DIRS}
    "${CMAKE_SOURCE_DIR}/renderer/include"
    "${CMAKE_SOURCE_DIR}/extern/entt/src"
    "${CMAKE_SOURCE_DIR}/extern/stduuid/include"
    "${CMAKE_SOURCE_DIR}/extern/stduuid"
    "${CMAKE_SOURCE_DIR}/extern/glm"
)
set(RENDERER_LINKED_LIBRARIES
    ${COMMON_LINKED_LIBRARIES}
    ${OPENGL_LIBRARIES}
    double-conversion
    EnTT::EnTT 
    stduuid
    common 
)
if(LAB_DEVICE STREQUAL "GPU")
    list(APPEND RENDERER_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
    list(APPEND RENDERER_LINKED_LIBRARIES ${CUDAToolkit_LIBRARIES})
endif()
add_subdirectory("renderer")

# lab
set(LAB_INCLUDE_DIRS
    ${RENDERER_INCLUDE_DIRS}
    "${CMAKE_SOURCE_DIR}/lab/include"
    "${CMAKE_SOURCE_DIR}/extern/dataframe/include"
)
set(LAB_LINKED_LIBRARIES
    ${RENDERER_LINKED_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${LZ4_LIBRARIES}
    nlohmann_json::nlohmann_json
    renderer
)
if(LAB_DEVICE STREQUAL "GPU")
    list(APPEND LAB_LINKED_LIBRARIES renderer_gpu)
endif()
add_subdirectory("lab")

# test
add_subdirectory("test")

# example
add_subdirectory("example")

