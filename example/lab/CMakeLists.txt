cmake_minimum_required(VERSION 3.25)

file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

add_executable(example main.cpp ${SRC_FILES} ${HEADER_FILES})
target_include_directories(example PUBLIC ${LAB_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(example PRIVATE ${LAB_LINKED_LIBRARIES} lab)
target_precompile_headers(example PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/main.h")
set_target_properties(example PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    COMPILE_FLAGS ${CMAKE_CXX_FLAGS} ${CMAKE_VEC_FLAGS} ${CMAKE_OPENMP_FLAGS}
    OUTPUT_NAME example
)

add_custom_command(TARGET example POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory output)

if(MSVC)
    include(copy_torch_dlls)
    copy_torch_dlls(example)
endif()