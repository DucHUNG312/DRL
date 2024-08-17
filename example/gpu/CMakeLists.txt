cmake_minimum_required(VERSION 3.25)

project(example_cu VERSION 1.0.0 LANGUAGES CXX CUDA)

enable_language(CUDA)

file(GLOB_RECURSE CUDA_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
)

file(GLOB_RECURSE HEADER_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
)

add_executable(${PROJECT_NAME} main.cu ${CUDA_FILES} ${HEADER_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC ${LAB_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LAB_LINKED_LIBRARIES} lab lab_gpu)
set_target_properties(${PROJECT_NAME} PROPERTIES 
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    COMPILE_FLAGS ${CMAKE_CUDA_FLAGS}
    OUTPUT_NAME ${PROJECT_NAME}
    CUDA_SEPARABLE_COMPILATION ON
)

add_custom_command(TARGET ${PROJECT_NAME}
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E make_directory output)

if(MSVC)
    include(copy_torch_dlls)
    copy_torch_dlls(${PROJECT_NAME})
endif()
