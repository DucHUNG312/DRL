cmake_minimum_required(VERSION 3.25)

project(test VERSION 1.0.0 LANGUAGES CXX)

enable_testing()

find_package(GTest REQUIRED)

file(GLOB_RECURSE SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

file(GLOB_RECURSE HEADER_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
)

set(TEST_INCLUDE_DIRS
    ${LAB_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set(TEST_LINKED_LIBRARIES
    ${LAB_LINKED_LIBRARIES}
    GTest::gtest_main
    lab
)

add_executable(${PROJECT_NAME} main.test.cpp ${SRC_FILES} ${HEADER_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC ${TEST_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${TEST_LINKED_LIBRARIES})
target_precompile_headers(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/main.h")
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    UNITY_BUILD ON UNITY_BUILD_MODE BATCH UNITY_BUILD_BATCH_SIZE 16
    OUTPUT_NAME ${PROJECT_NAME}
)

add_custom_command(TARGET ${PROJECT_NAME}
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E make_directory output)

if(MSVC)
    include(copy_torch_dlls)
    copy_torch_dlls(${PROJECT_NAME})
endif()