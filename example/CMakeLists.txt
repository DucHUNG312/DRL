add_subdirectory("lab")

if(LAB_DEVICE STREQUAL "GPU")
    add_subdirectory("gpu")
endif()

