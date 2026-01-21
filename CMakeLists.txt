cmake_minimum_required(VERSION 3.15.0)
project(python-blosc2)
# Specifying Python version below is tricky, but if you don't specify the minimum version here,
# it would not consider python3 when looking for the executable. This is problematic since Fedora
# does not include a python symbolic link to python3.
# find_package(Python 3.12 COMPONENTS Interpreter NumPy Development.Module REQUIRED)
# IMO, this would need to be solved in Fedora, so we can just use the following line:
find_package(Python COMPONENTS Interpreter NumPy Development.Module REQUIRED)

# Add custom command to generate the version file
add_custom_command(
  OUTPUT src/blosc2/version.py
  COMMAND ${Python_EXECUTABLE} generate_version.py
  DEPENDS generate_version.py pyproject.toml
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)

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

if(DEFINED ENV{USE_SYSTEM_BLOSC2})
    set(USE_SYSTEM_BLOSC2 ON)
endif()

if(USE_SYSTEM_BLOSC2)
    set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(Blosc2 REQUIRED IMPORTED_TARGET blosc2)
    target_link_libraries(blosc2_ext PRIVATE PkgConfig::Blosc2)
else()
    set(STATIC_LIB ON CACHE BOOL "Build a static version of the blosc library.")
    set(SHARED_LIB ON CACHE BOOL "Build a shared library version of the blosc library.")
    set(BUILD_TESTS OFF CACHE BOOL "Build C-Blosc2 tests")
    set(BUILD_EXAMPLES OFF CACHE BOOL "Build C-Blosc2 examples")
    set(BUILD_BENCHMARKS OFF CACHE BOOL "Build C-Blosc2 benchmarks")
    set(BUILD_FUZZERS OFF CACHE BOOL "Build C-Blosc2 fuzzers")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_INSTALL_INCLUDEDIR ${SKBUILD_PLATLIB_DIR}/blosc2/include) # directory for include files
    set(CMAKE_INSTALL_LIBDIR ${SKBUILD_PLATLIB_DIR}/blosc2/lib) # directory for libblosc2 and pkgconfig
    set(Blosc2_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/blosc2) # directory for cmake files
    set(CMAKE_INSTALL_BINDIR ${SKBUILD_PLATLIB_DIR}/blosc2/lib) # directory for libblosc2.dll on windows
    # we will put the binaries of the C-Blosc2 library into the wheels according to PEP
    set(BLOSC_INSTALL ON)
    include(FetchContent)
    FetchContent_Declare(blosc2
        GIT_REPOSITORY https://github.com/Blosc/c-blosc2
        GIT_TAG 2effe806e318f4d474705460b0a9a172b8e07ff9  # v2.22.0
    )
    FetchContent_MakeAvailable(blosc2)
    include_directories("${blosc2_SOURCE_DIR}/include")
    target_link_libraries(blosc2_ext PRIVATE blosc2_static)
endif()

# TODO
# CHECK THIS
if(UNIX)
  set_target_properties(blosc2_ext PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$<IF:$<PLATFORM_ID:Darwin>,@loader_path/lib,\$ORIGIN/lib>"
  )
endif()

if(WIN32)
  add_custom_command(TARGET blosc2_ext POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:blosc2_shared>
            $<TARGET_FILE_DIR:blosc2_ext>
  )
endif()

# Python extension -> site-packages/blosc2
install(
  TARGETS blosc2_ext
  LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/blosc2
)
