set(example_name "example1")

add_executable(${example_name})

set(CXX_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
)

set(HEADER_FILES

)

target_sources(${example_name}
    PRIVATE ${CXX_SRCS}
)

target_include_directories(${example_name} 
    PUBLIC 
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lab/include>"
        "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(${example_name} 
    PUBLIC 
        lab
)

# Install
install(FILES ${HEADER_FILES} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/lab/examples/example1")