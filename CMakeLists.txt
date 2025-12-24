cmake_minimum_required(VERSION 3.15.0)

if(WIN32)
    cmake_policy(SET CMP0091 NEW)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()


if(WIN32)
    if(CMAKE_GENERATOR MATCHES "Visual Studio")
        if(NOT DEFINED CMAKE_GENERATOR_TOOLSET)
            set(CMAKE_GENERATOR_TOOLSET "ClangCL" CACHE STRING "Use ClangCL toolset for C99 support on Windows." FORCE)
        endif()
    else()
        if(NOT DEFINED CMAKE_C_COMPILER AND NOT DEFINED ENV{CC})
            set(CMAKE_C_COMPILER clang-cl CACHE STRING "Use clang-cl for C99 support on Windows." FORCE)
        endif()
        if(NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
            set(CMAKE_CXX_COMPILER clang-cl CACHE STRING "Use clang-cl for C99 support on Windows." FORCE)
        endif()
    endif()
endif()

project(python-blosc2)

if(WIN32 AND NOT CMAKE_C_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "Windows builds require clang-cl for C99 support. Set CC/CXX to clang-cl or use the ClangCL toolset.")
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
Python_add_library(blosc2_ext MODULE blosc2_ext.c
                   "${CMAKE_CURRENT_SOURCE_DIR}/src/blosc2/miniexpr.c" WITH_SOABI)
# We need to link against NumPy
target_link_libraries(blosc2_ext PRIVATE Python::NumPy)

# Add include directory for miniexpr.h and others
target_include_directories(blosc2_ext PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/blosc2")
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
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    # we want the binaries of the C-Blosc2 library to go into the wheels
    set(BLOSC_INSTALL ON)
    include(FetchContent)
    FetchContent_Declare(blosc2
        GIT_REPOSITORY https://github.com/Blosc/c-blosc2
        GIT_TAG 290853dc15dc0e0e887cc72fbaac692cefd75014  # openzl
        # in case you want to use a local copy of c-blosc2 for development, uncomment the line below
        # SOURCE_DIR "/Users/faltet/blosc/c-blosc2"
    )
    FetchContent_MakeAvailable(blosc2)
    include_directories("${blosc2_SOURCE_DIR}/include")
    target_link_libraries(blosc2_ext PRIVATE blosc2_static)
    if(WIN32 AND CMAKE_C_COMPILER_ID STREQUAL "Clang")
        set(_blosc2_ssse3_flag "-mssse3")
        if(CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
            set(_blosc2_ssse3_flag "/clang:-mssse3")
        endif()
        if(TARGET blosc2_static)
            target_compile_options(blosc2_static PRIVATE "${_blosc2_ssse3_flag}")
        endif()
        unset(_blosc2_ssse3_flag)
    endif()
endif()

add_custom_command(
    TARGET blosc2_ext POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:blosc2_ext> ${CMAKE_SOURCE_DIR}/blosc2
)

install(TARGETS blosc2_ext LIBRARY DESTINATION blosc2)
