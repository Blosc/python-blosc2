cmake_minimum_required(VERSION 3.15.0)

if(WIN32)
    cmake_policy(SET CMP0091 NEW)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" CACHE STRING "" FORCE)
endif()


if(WIN32 AND CMAKE_GENERATOR MATCHES "Visual Studio")
    if(NOT DEFINED CMAKE_GENERATOR_TOOLSET)
        set(CMAKE_GENERATOR_TOOLSET "ClangCL" CACHE STRING "Use ClangCL toolset for C99/C11 support on Windows." FORCE)
    endif()
endif()

project(python-blosc2)

if(WIN32 AND NOT CMAKE_C_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "Windows builds require clang-cl. Set CC/CXX to clang-cl or configure CMake with -T ClangCL.")
endif()
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

# Fetch and build miniexpr library
include(FetchContent)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(MINIEXPR_BUILD_SHARED OFF CACHE BOOL "Build miniexpr shared library" FORCE)
set(MINIEXPR_BUILD_TESTS OFF CACHE BOOL "Build miniexpr tests" FORCE)
set(MINIEXPR_BUILD_EXAMPLES OFF CACHE BOOL "Build miniexpr examples" FORCE)
set(MINIEXPR_BUILD_BENCH OFF CACHE BOOL "Build miniexpr benchmarks" FORCE)

FetchContent_Declare(miniexpr
    GIT_REPOSITORY https://github.com/Blosc/miniexpr.git
    #GIT_TAG ndim  # latest me_compile_nd()/me_eval_nd() APIs
    GIT_TAG 8c50850094e156ce568186edef667fabecbd00ff  # latest commit in ndim
    # In case you want to use a local copy of miniexpr for development, uncomment the line below
    # SOURCE_DIR "/Users/faltet/blosc/miniexpr"
)
FetchContent_MakeAvailable(miniexpr)

# Link against miniexpr static library
target_link_libraries(blosc2_ext PRIVATE miniexpr_static)

target_compile_features(blosc2_ext PRIVATE c_std_11)
if(WIN32 AND CMAKE_C_COMPILER_ID STREQUAL "Clang")
    execute_process(
        COMMAND "${CMAKE_C_COMPILER}" -print-resource-dir
        OUTPUT_VARIABLE _clang_resource_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_clang_resource_dir)
        if(CMAKE_SIZEOF_VOID_P EQUAL 8)
            set(_clang_builtins "${_clang_resource_dir}/lib/windows/clang_rt.builtins-x86_64.lib")
        else()
            set(_clang_builtins "${_clang_resource_dir}/lib/windows/clang_rt.builtins-i386.lib")
        endif()
        if(EXISTS "${_clang_builtins}")
            target_link_libraries(blosc2_ext PRIVATE "${_clang_builtins}")
        endif()
        unset(_clang_builtins)
    endif()
    unset(_clang_resource_dir)
endif()

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
    if(DEFINED ENV{DEACTIVATE_OPENZL})
        set(DEACTIVATE_OPENZL ON CACHE BOOL "Do not include support for the OpenZL library.")
    endif()
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    # we want the binaries of the C-Blosc2 library to go into the wheels
    set(BLOSC_INSTALL ON)
    include(FetchContent)
    FetchContent_Declare(blosc2
        GIT_REPOSITORY https://github.com/Blosc/c-blosc2
        GIT_TAG 011c9e537f28299c536294d842e1a3d0e41db24f  # openzl_plugin + miniexpr
        # GIT_TAG main
        GIT_SHALLOW TRUE  # fetch only the latest commit (only works with a branch in GIT_TAG)
        # in case you want to use a local copy of c-blosc2 for development, uncomment the line below
        # SOURCE_DIR "/Users/faltet/blosc/c-blosc2"
    )
    FetchContent_MakeAvailable(blosc2)
    include_directories("${blosc2_SOURCE_DIR}/include")
    target_link_libraries(blosc2_ext PRIVATE blosc2_static)
endif()

add_custom_command(
    TARGET blosc2_ext POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:blosc2_ext> ${CMAKE_SOURCE_DIR}/blosc2
)

install(TARGETS blosc2_ext LIBRARY DESTINATION blosc2)
