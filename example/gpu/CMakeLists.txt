cmake_minimum_required(VERSION 3.25)

enable_language(CUDA)

cmake_minimum_required(VERSION 3.25)

file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
file(GLOB_RECURSE HEADER_FILES${CMAKE_CURRENT_SOURCE_DIR}/*.h)

add_executable(example_gpu main.cu ${SRC_FILES} ${HEADER_FILES})

target_include_directories(example_gpu PUBLIC ${LAB_INCLUDE_DIRS})

target_link_libraries(example_gpu PRIVATE ${LAB_LINKED_LIBRARIES} lab)

set_target_properties(example_gpu PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    COMPILE_FLAGS ${CMAKE_CUDA_FLAGS}
    OUTPUT_NAME ${PROJECT_NAME}
    CUDA_SEPARABLE_COMPILATION ON
)

add_custom_command(TARGET example_gpu POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory output)

if(MSVC)
    include(copy_torch_dlls)
    copy_torch_dlls(example_gpu)
endif()