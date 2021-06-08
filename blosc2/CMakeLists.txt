set(STATIC_LIB ON CACHE BOOL "Build a static version of the blosc library.")
set(SHARED_LIB ON CACHE BOOL "Build a shared library version of the blosc
    library.")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# we want the binaries of the C-Blosc2 library to go into the wheels
set(BLOSC_INSTALL ON)
add_subdirectory(c-blosc2)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/c-blosc2/blosc")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/c-blosc2/include")


add_cython_target(blosc2_ext blosc2_ext.pyx)
add_library(blosc2_ext MODULE ${blosc2_ext})

target_link_libraries(blosc2_ext blosc2_static)

python_extension_module(blosc2_ext)

add_custom_command(
    TARGET blosc2_ext POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:blosc2_ext> ${CMAKE_SOURCE_DIR}/blosc2
)

install(TARGETS blosc2_ext LIBRARY DESTINATION blosc2)
