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

if(LAB_DEVICE STREQUAL "GPU")
    list(APPEND TEST_LINKED_LIBRARIES lab_gpu)
endif()

add_executable(${PROJECT_NAME} main.test.cpp ${SRC_FILES} ${HEADER_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC ${TEST_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${TEST_LINKED_LIBRARIES})

set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    OUTPUT_NAME ${PROJECT_NAME}
)

add_custom_command(TARGET ${PROJECT_NAME}
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E make_directory output)

if(MSVC)
    include(copy_torch_dlls)
    copy_torch_dlls(${PROJECT_NAME})
endif()