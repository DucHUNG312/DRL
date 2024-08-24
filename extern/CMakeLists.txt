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

#lz4
FetchContent_Declare(
    extern_lz4
    GIT_REPOSITORY https://github.com/lz4/lz4.git
    GIT_TAG v1.10.0
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lz4
)
FetchContent_MakeAvailable(extern_lz4)

#json
FetchContent_Declare(
    extern_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/json
)
FetchContent_MakeAvailable(extern_json)

#DataFrame
FetchContent_Declare(
    extern_dataframe
    GIT_REPOSITORY https://github.com/hosseinmoein/DataFrame.git
    GIT_TAG 2.1.0
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dataframe
)
FetchContent_MakeAvailable(extern_dataframe)

# double-conversion
FetchContent_Declare(
    extern_double_conversion
    GIT_REPOSITORY https://github.com/google/double-conversion.git
    GIT_TAG v3.3.0
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/double_conversion
)
FetchContent_MakeAvailable(extern_double_conversion)

# entt
FetchContent_Declare(
    extern_entt
    GIT_REPOSITORY https://github.com/skypjack/entt.git
    GIT_TAG v3.13.2
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/entt
)
FetchContent_MakeAvailable(extern_entt)

# stduuid
FetchContent_Declare(
    extern_stduuid
    GIT_REPOSITORY https://github.com/mariusbancila/stduuid.git
    GIT_TAG v1.2.3
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/stduuid
)
set(UUID_SYSTEM_GENERATOR ON CACHE BOOL "Use system generator" FORCE)

FetchContent_MakeAvailable(extern_stduuid)