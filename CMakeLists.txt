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

if(EMSCRIPTEN)
  set(MINIEXPR_ENABLE_TCC_JIT ON CACHE BOOL "Enable TCC JIT in Emscripten builds" FORCE)
  set(MINIEXPR_WASM32_SIDE_MODULE ON CACHE BOOL "Use host-registered wasm32 JIT helpers" FORCE)
endif()

FetchContent_Declare(miniexpr
    GIT_REPOSITORY https://github.com/Blosc/miniexpr.git
    GIT_TAG 393c373a0f02735784aa7afe767eb310ebe99713
    # SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../miniexpr
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
        GIT_TAG 5b7d426b07e5be328b5accb88444d5e7aeabce53 #v2.23.0
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
  if(TARGET blosc2_shared)
    add_custom_command(TARGET blosc2_ext POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              $<TARGET_FILE:blosc2_shared>
              $<TARGET_FILE_DIR:blosc2_ext>
    )
  endif()
endif()

# Python extension -> site-packages/blosc2
install(
  TARGETS blosc2_ext
  LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/blosc2
)

install(
  FILES "${miniexpr_SOURCE_DIR}/src/me_jit_glue.js"
  DESTINATION ${SKBUILD_PLATLIB_DIR}/blosc2
)

# Install bundled libtcc next to the Python package (separate LGPL artifact).
if(MINIEXPR_ENABLE_TCC_JIT)
  if(APPLE)
    install(
      FILES "${miniexpr_BINARY_DIR}/libtcc.dylib"
      DESTINATION ${SKBUILD_PLATLIB_DIR}/blosc2/lib
      OPTIONAL
    )
  elseif(WIN32)
    install(
      FILES
        "${miniexpr_BINARY_DIR}/tcc.dll"
        "${miniexpr_BINARY_DIR}/Debug/tcc.dll"
        "${miniexpr_BINARY_DIR}/Release/tcc.dll"
        "${miniexpr_BINARY_DIR}/RelWithDebInfo/tcc.dll"
        "${miniexpr_BINARY_DIR}/MinSizeRel/tcc.dll"
      DESTINATION ${SKBUILD_PLATLIB_DIR}/blosc2/lib
      OPTIONAL
    )
  else()
    install(
      FILES
        "${miniexpr_BINARY_DIR}/libtcc.so"
        "${miniexpr_BINARY_DIR}/libtcc.so.1"
      DESTINATION ${SKBUILD_PLATLIB_DIR}/blosc2/lib
      OPTIONAL
    )
  endif()
endif()
