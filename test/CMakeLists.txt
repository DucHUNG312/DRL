set(TEST_NAME "test")

enable_testing()

find_package(GTest REQUIRED)

add_executable(${TEST_NAME})

file(GLOB_RECURSE 
    SRC_FILES 
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

file(GLOB_RECURSE 
    HEADER_FILES 
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
)

target_sources(${TEST_NAME} 
    PRIVATE
        ${SRC_FILES}
)

target_include_directories(${TEST_NAME} 
    PUBLIC 
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lab/include>"
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(${TEST_NAME} 
    PUBLIC 
        lab
        GTest::gtest_main
)

# Install
install(FILES ${HEADER_FILES} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/test")