cmake_minimum_required(VERSION 3.18)

project(example VERSION 1.0.0 LANGUAGES CXX)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

file(GLOB_RECURSE SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

file(GLOB_RECURSE CUDA_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
)

file(GLOB_RECURSE HEADER_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
)

include_directories(
    ${LAB_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME} main.cpp ${SRC_FILES} ${CUDA_FILES} ${HEADER_FILES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LAB_LINKED_LIBRARIES} lab)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    COMPILE_FLAGS ${CMAKE_CXX_FLAGS} ${CMAKE_VEC_FLAGS} ${CMAKE_OPENMP_FLAGS}
    OUTPUT_NAME ${PROJECT_NAME}
    CUDA_SEPARABLE_COMPILATION ON
)
if(LAB_DEVICE STREQUAL "GPU")
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS}>)
    target_compile_definitions(${PROJECT_NAME} PRIVATE LAB_GPU_BUILD)
endif()

add_custom_command(TARGET ${PROJECT_NAME}
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E make_directory output)

if(MSVC)
    include(copy_torch_dlls)
    copy_torch_dlls(${PROJECT_NAME})
endif()
