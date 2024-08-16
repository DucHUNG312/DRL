cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(RL VERSION 1.0.0 LANGUAGES CXX)

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

find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

if(LAB_DEVICE STREQUAL "GPU")
    if (NOT CUDA_FOUND)
        message(WARNING "GPU build is set but CUDA toolkit not found. Use CPU instead.")
        set(LAB_DEVICE "CPU")
    else()
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURE)
            set(CMAKE_CUDA_ARCHITECTURE "52")
            message(WARNING "CMAKE_CUDA_ARCHITECTURE is not set. Use default architecture.")
        endif()
        message(STATUS "CMAKE_CUDA_ARCHITECTURE: ${CMAKE_CUDA_ARCHITECTURE}")
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
set(PYTORCH_VERSION "2.3.0")
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
#find_package(OpenCV REQUIRED)
find_package(GLUT)
find_package(OpenGL)
find_package(Eigen3 REQUIRED)
find_package(LZ4 REQUIRED)

set(LAB_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(LAB_INCLUDE_DIRS
    ${LAB_INCLUDE_DIR}
)
set(LAB_LINKED_LIBRARIES
    spdlog::spdlog_header_only
    ${TORCH_LIBRARIES}
    #${OpenCV_LIBS}
    ${GLUT_LIBRARIES} 
    ${OPENGL_LIBRARIES}
    Eigen3::Eigen
    ${LZ4_LIBRARIES}
)
if(LAB_DEVICE STREQUAL "GPU")
    list(APPEND LAB_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
    list(APPEND LAB_LINKED_LIBRARIES ${CUDA_LIBRARIES})
endif()

add_subdirectory("src")
add_subdirectory("test")
add_subdirectory("example")

