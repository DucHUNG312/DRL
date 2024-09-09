# libtorch
find_package(Torch REQUIRED)
find_package(Threads REQUIRED)

add_definitions(-DTORCH_VERSION_MAJOR=${Torch_VERSION_MAJOR})
add_definitions(-DTORCH_VERSION_MINOR=${Torch_VERSION_MINOR})
add_definitions(-DTORCH_VERSION_PATCH=${Torch_VERSION_PATCH})

# spdlog
if(NOT TARGET spdlog)
    find_package(spdlog REQUIRED)
endif()

# nlohmann_json
if(NOT TARGET nlohmann_json)
    find_package(nlohmann_json REQUIRED)
endif()