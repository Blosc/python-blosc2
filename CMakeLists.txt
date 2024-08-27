cmake_minimum_required(VERSION 3.15.0)
project(python-blosc2)
# Specifying Python version below is tricky, but if you don't specify the minimum version here,
# it would not consider python3 when looking for the executable. This is problematic since Fedora
# does not include a python symbolic link to python3.
find_package(Python 3.10 COMPONENTS Interpreter NumPy Development.Module REQUIRED)

# Compile the Cython extension manually...
add_custom_command(
  OUTPUT blosc2_ext.c
  COMMAND Python::Interpreter -m cython
          "${CMAKE_CURRENT_SOURCE_DIR}/src/blosc2/blosc2_ext.pyx" --output-file blosc2_ext.c
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/blosc2/blosc2_ext.pyx"
  VERBATIM)
# ...and add it to the target
Python_add_library(blosc2_ext MODULE blosc2_ext.c WITH_SOABI)
# We need to link against NumPy
target_link_libraries(blosc2_ext PRIVATE Python::NumPy)

if(USE_SYSTEM_BLOSC2)
    set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(Blosc2 REQUIRED IMPORTED_TARGET blosc2)
    target_link_libraries(blosc2_ext PkgConfig::Blosc2)
else()
    set(STATIC_LIB ON CACHE BOOL "Build a static version of the blosc library.")
    set(SHARED_LIB ON CACHE BOOL "Build a shared library version of the blosc library.")
    set(BUILD_TESTS OFF CACHE BOOL "Build C-Blosc2 tests")
    set(BUILD_EXAMPLES OFF CACHE BOOL "Build C-Blosc2 examples")
    set(BUILD_BENCHMARKS OFF CACHE BOOL "Build C-Blosc2 benchmarks")
    set(BUILD_FUZZERS OFF CACHE BOOL "Build C-Blosc2 fuzzers")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    # we want the binaries of the C-Blosc2 library to go into the wheels
    set(BLOSC_INSTALL ON)
    add_subdirectory(${PROJECT_SOURCE_DIR}/c-blosc2)
    include_directories("${PROJECT_SOURCE_DIR}/c-blosc2/include")
    target_link_libraries(blosc2_ext PRIVATE blosc2_static)
endif()

add_custom_command(
    TARGET blosc2_ext POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:blosc2_ext> ${CMAKE_SOURCE_DIR}/blosc2
)

install(TARGETS blosc2_ext LIBRARY DESTINATION blosc2)
