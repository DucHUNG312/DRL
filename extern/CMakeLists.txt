cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(extern VERSION 1.0.0 LANGUAGES CXX)

include(FetchContent)

# spdlog
FetchContent_Declare(
    extern_spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.9.2
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/spdlog
)

set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "Set to ON to build examples" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "Set to ON to build tests" FORCE)
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "Set to ON to the build bench" FORCE)
set(SPDLOG_INSTALL OFF CACHE BOOL "Set to ON to generate the install target" FORCE)

FetchContent_MakeAvailable(extern_spdlog)

# gtest
FetchContent_Declare(
    extern_googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.0
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/googletest
)

set(BUILD_GMOCK ON CACHE BOOL "Set to OFF to do not build googlemock subproject" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "Set to ON to generate the install target" FORCE)
set(gtest_build_samples OFF CACHE BOOL "Set to ON to build examples" FORCE)
set(gtest_build_tests OFF CACHE BOOL "Set to ON to build tests" FORCE)

FetchContent_MakeAvailable(extern_googletest)

# opencv
FetchContent_Declare(
    extern_opencv
    GIT_REPOSITORY https://github.com/opencv/opencv
    GIT_TAG 4.6.0
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/opencv
)

FetchContent_MakeAvailable(extern_opencv)

# eigen
FetchContent_Declare(
    extern_eigen
    GIT_REPOSITORY https://github.com/DucHUNG312/eigen
    GIT_TAG 3.4.0
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/eigen
)
set(EIGEN_BUILD_BTL OFF CACHE BOOL "Set to ON to build benchmark suite" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "Set to ON to build Eigen documentation" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Set to ON to enable creation of Eigen tests" FORCE)
FetchContent_MakeAvailable(extern_eigen)
