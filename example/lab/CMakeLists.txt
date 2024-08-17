cmake_minimum_required(VERSION 3.25)

project(example VERSION 1.0.0 LANGUAGES CXX)

file(GLOB_RECURSE SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

file(GLOB_RECURSE HEADER_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
)

set(EXAMPLE_LINKED_LIBRARIES
    ${LAB_LINKED_LIBRARIES} 
    lab
)

if(LAB_DEVICE STREQUAL "GPU")
    list(APPEND EXAMPLE_LINKED_LIBRARIES lab_gpu)
endif()

add_executable(${PROJECT_NAME} main.cpp ${SRC_FILES} ${HEADER_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC ${LAB_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${EXAMPLE_LINKED_LIBRARIES})
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    COMPILE_FLAGS ${CMAKE_CXX_FLAGS} ${CMAKE_VEC_FLAGS} ${CMAKE_OPENMP_FLAGS}
    OUTPUT_NAME ${PROJECT_NAME}
)

add_custom_command(TARGET ${PROJECT_NAME}
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E make_directory output)

if(MSVC)
    include(copy_torch_dlls)
    copy_torch_dlls(${PROJECT_NAME})
endif()
