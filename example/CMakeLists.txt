cmake_minimum_required(VERSION 3.25)

project(example VERSION 1.0.0 LANGUAGES CXX)

add_subdirectory("renderer")
add_subdirectory("lab")
# if(LAB_DEVICE STREQUAL "GPU")
#     add_subdirectory("gpu")
# endif()

