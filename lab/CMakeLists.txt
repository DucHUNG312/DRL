set(LIBRARY_NAME "lab")

file(GLOB_RECURSE SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

file(GLOB_RECURSE HEADER_FILES
    "${CMAKE_SOURCE_DIR}/lab/include/lab/*.h"
)

add_library(${LIBRARY_NAME} SHARED ${SRC_FILES} ${HEADER_FILES})

target_include_directories(${LIBRARY_NAME} 
    PUBLIC 
        "${CMAKE_SOURCE_DIR}/lab/include/"
)

target_link_libraries(${LIBRARY_NAME} 
    PUBLIC
        ${TORCH_LIBRARIES}
        nlohmann_json::nlohmann_json
)

if(LAB_DEVICE STREQUAL "GPU")
    target_include_directories(${LIBRARY_NAME} 
        PUBLIC 
            ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(${LIBRARY_NAME}  
        PUBLIC
            ${CUDAToolkit_LIBRARIES}
    )
endif()

target_precompile_headers(${LIBRARY_NAME} 
    PUBLIC 
        "${CMAKE_SOURCE_DIR}/lab/include/lab/common/common.h"
)

# Install
install(FILES ${HEADER_FILES} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/lab")