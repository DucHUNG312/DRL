# Find CUDA.
find_package(CUDAToolkit REQUIRED)

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_C_COMPILER}")
endif()
enable_language(CUDA)
if("X${CMAKE_CUDA_STANDARD}" STREQUAL "X" )
  set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
endif()
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CMP0074 - find_package will respect <PackageName>_ROOT variables
cmake_policy(PUSH)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.12.0)
  cmake_policy(SET CMP0074 NEW)
endif()

find_package(CUDAToolkit REQUIRED)

cmake_policy(POP)

if(NOT CMAKE_CUDA_COMPILER_VERSION VERSION_EQUAL CUDAToolkit_VERSION)
  message(FATAL_ERROR "Found two conflicting CUDA versions:\n"
                      "V${CMAKE_CUDA_COMPILER_VERSION} in '${CUDA_INCLUDE_DIRS}' and\n"
                      "V${CUDAToolkit_VERSION} in '${CUDAToolkit_INCLUDE_DIRS}'")
endif()

if(NOT TARGET CUDA::nvToolsExt)
  message(FATAL_ERROR "Failed to find nvToolsExt")
endif()

message(STATUS "DRL: CUDA detected: " ${CUDA_VERSION})
message(STATUS "DRL: CUDA nvcc is: " ${CUDA_NVCC_EXECUTABLE})
message(STATUS "DRL: CUDA toolkit directory: " ${CUDA_TOOLKIT_ROOT_DIR})
if(CUDA_VERSION VERSION_LESS 12.0)
  message(FATAL_ERROR "DRL requires CUDA 12.0 or above.")
endif()

if(CUDA_FOUND)
  # Sometimes, we may mismatch nvcc with the CUDA headers we are
  # compiling with, e.g., if a ccache nvcc is fed to us by CUDA_NVCC_EXECUTABLE
  # but the PATH is not consistent with CUDA_HOME.  It's better safe
  # than sorry: make sure everything is consistent.
  if(MSVC AND CMAKE_GENERATOR MATCHES "Visual Studio")
    # When using Visual Studio, it attempts to lock the whole binary dir when
    # `try_run` is called, which will cause the build to fail.
    string(RANDOM BUILD_SUFFIX)
    set(PROJECT_RANDOM_BINARY_DIR "${PROJECT_BINARY_DIR}/${BUILD_SUFFIX}")
  else()
    set(PROJECT_RANDOM_BINARY_DIR "${PROJECT_BINARY_DIR}")
  endif()
  set(file "${PROJECT_BINARY_DIR}/detect_cuda_version.cc")
  file(WRITE ${file} ""
    "#include <cuda.h>\n"
    "#include <cstdio>\n"
    "int main() {\n"
    "  printf(\"%d.%d\", CUDA_VERSION / 1000, (CUDA_VERSION / 10) % 100);\n"
    "  return 0;\n"
    "}\n"
    )
  if(NOT CMAKE_CROSSCOMPILING)
    try_run(run_result compile_result ${PROJECT_RANDOM_BINARY_DIR} ${file}
      CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CUDA_INCLUDE_DIRS}"
      LINK_LIBRARIES ${CUDA_LIBRARIES}
      RUN_OUTPUT_VARIABLE cuda_version_from_header
      COMPILE_OUTPUT_VARIABLE output_var
      )
    if(NOT compile_result)
      message(FATAL_ERROR "DRL: Couldn't determine version from header: " ${output_var})
    endif()
    message(STATUS "DRL: Header version is: " ${cuda_version_from_header})
    if(NOT cuda_version_from_header STREQUAL ${CUDA_VERSION_STRING})
      # Force CUDA to be processed for again next time
      # TODO: I'm not sure if this counts as an implementation detail of
      # FindCUDA
      set(${cuda_version_from_findcuda} ${CUDA_VERSION_STRING})
      unset(CUDA_TOOLKIT_ROOT_DIR_INTERNAL CACHE)
      # Not strictly necessary, but for good luck.
      unset(CUDA_VERSION CACHE)
      # Error out
      message(FATAL_ERROR "FindCUDA says CUDA version is ${cuda_version_from_findcuda} (usually determined by nvcc), "
        "but the CUDA headers say the version is ${cuda_version_from_header}.  This often occurs "
        "when you set both CUDA_HOME and CUDA_NVCC_EXECUTABLE to "
        "non-standard locations, without also setting PATH to point to the correct nvcc.  "
        "Perhaps, try re-running this command again with PATH=${CUDA_TOOLKIT_ROOT_DIR}/bin:$PATH.  ")
    endif()
  endif()
endif()

# ---[ CUDA libraries wrapper

# find libcuda.so and lbnvrtc.so
# For libcuda.so, we will find it under lib, lib64, and then the
# stubs folder, in case we are building on a system that does not
# have cuda driver installed. On windows, we also search under the
# folder lib/x64.
set(CUDA_CUDA_LIB "${CUDA_cuda_driver_LIBRARY}" CACHE FILEPATH "")
set(CUDA_NVRTC_LIB "${CUDA_nvrtc_LIBRARY}" CACHE FILEPATH "")
if(CUDA_NVRTC_LIB AND NOT CUDA_NVRTC_SHORTHASH)
  if("${Python_EXECUTABLE}" STREQUAL "")
    set(_python_exe "python3")
  else()
    set(_python_exe "${Python_EXECUTABLE}")
  endif()
  execute_process(
    COMMAND "${_python_exe}" -c
    "import hashlib;hash=hashlib.sha256();hash.update(open('${CUDA_NVRTC_LIB}','rb').read());print(hash.hexdigest()[:8])"
    RESULT_VARIABLE _retval
    OUTPUT_VARIABLE CUDA_NVRTC_SHORTHASH)
  if(NOT _retval EQUAL 0)
    message(WARNING "Failed to compute shorthash for libnvrtc.so")
    set(CUDA_NVRTC_SHORTHASH "XXXXXXXX")
  else()
    string(STRIP "${CUDA_NVRTC_SHORTHASH}" CUDA_NVRTC_SHORTHASH)
    message(STATUS "${CUDA_NVRTC_LIB} shorthash is ${CUDA_NVRTC_SHORTHASH}")
  endif()
endif()

# Add onnx namepsace definition to nvcc
if(ONNX_NAMESPACE)
  list(APPEND CUDA_NVCC_FLAGS "-DONNX_NAMESPACE=${ONNX_NAMESPACE}")
else()
  list(APPEND CUDA_NVCC_FLAGS "-DONNX_NAMESPACE=onnx_c2")
endif()

# Don't activate VC env again for Ninja generators with MSVC on Windows if CUDAHOSTCXX is not defined
# by adding --use-local-env.
if(MSVC AND CMAKE_GENERATOR STREQUAL "Ninja" AND NOT DEFINED ENV{CUDAHOSTCXX})
  list(APPEND CUDA_NVCC_FLAGS "--use-local-env")
endif()

# setting nvcc arch flags
cuda_get_nvcc_gencode_flag(NVCC_FLAGS_EXTRA)
# CMake 3.18 adds integrated support for architecture selection, but we can't rely on it
set(CMAKE_CUDA_ARCHITECTURES OFF)
list(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})
message(STATUS "Added CUDA NVCC flags for: ${NVCC_FLAGS_EXTRA}")

# disable some nvcc diagnostic that appears in boost, glog, glags, opencv, etc.
foreach(diag cc_clobber_ignored
             field_without_dll_interface
             base_class_has_different_dll_interface
             dll_interface_conflict_none_assumed
             dll_interface_conflict_dllexport_assumed
             bad_friend_decl)
  list(APPEND SUPPRESS_WARNING_FLAGS --diag_suppress=${diag})
endforeach()
string(REPLACE ";" "," SUPPRESS_WARNING_FLAGS "${SUPPRESS_WARNING_FLAGS}")
list(APPEND CUDA_NVCC_FLAGS -Xcudafe ${SUPPRESS_WARNING_FLAGS})

set(CUDA_PROPAGATE_HOST_FLAGS_BLOCKLIST "-Werror")
if(MSVC)
  list(APPEND CUDA_NVCC_FLAGS "--Werror" "cross-execution-space-call")
  list(APPEND CUDA_NVCC_FLAGS "--no-host-device-move-forward")
endif()

# Debug and Release symbol support
if(MSVC)
    string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -Xcompiler /MDd")
    string(APPEND CMAKE_CUDA_FLAGS_MINSIZEREL " -Xcompiler /MD")
    string(APPEND CMAKE_CUDA_FLAGS_RELEASE " -Xcompiler /MD")
    string(APPEND CMAKE_CUDA_FLAGS_RELWITHDEBINFO " -Xcompiler /MD")
  if(CUDA_NVCC_FLAGS MATCHES "Zi")
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "-FS")
  endif()
elseif(CUDA_DEVICE_DEBUG)
  list(APPEND CUDA_NVCC_FLAGS "-g" "-G")  # -G enables device code debugging symbols
endif()

# Set expt-relaxed-constexpr to suppress Eigen warnings
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

# Set expt-extended-lambda to support lambda on device
list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda")

foreach(FLAG ${CUDA_NVCC_FLAGS})
  string(FIND "${FLAG}" " " flag_space_position)
  if(NOT flag_space_position EQUAL -1)
    message(FATAL_ERROR "Found spaces in CUDA_NVCC_FLAGS entry '${FLAG}'")
  endif()
  string(APPEND CMAKE_CUDA_FLAGS " ${FLAG}")
endforeach()