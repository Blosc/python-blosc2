find_program(CYTHON "cython")

add_custom_command(
  OUTPUT blosc2_ext.c
  DEPENDS blosc2_ext.pyx
  VERBATIM
  COMMAND "${CYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/blosc2_ext.pyx" --output-file
          "${CMAKE_CURRENT_BINARY_DIR}/blosc2_ext.c")

Python_add_library(blosc2_ext MODULE "${CMAKE_CURRENT_BINARY_DIR}/blosc2_ext.c" WITH_SOABI)

if(USE_SYSTEM_BLOSC2)
    set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(Blosc2 REQUIRED blosc2)
    target_include_directories(blosc2_ext PRIVATE ${Blosc2_INCLUDE_DIRS})
    target_link_libraries(blosc2_ext ${Blosc2_LDFLAGS})
else()
    set(STATIC_LIB ON CACHE BOOL "Build a static version of the blosc library.")
    set(SHARED_LIB ON CACHE BOOL "Build a shared library version of the blosc library.")
    set(BUILD_TESTS OFF CACHE BOOL "Build C-Blosc2 tests")
    set(BUILD_EXAMPLES OFF CACHE BOOL "Build C-Blosc2 examples")
    set(BUILD_BENCHMARKS OFF CACHE BOOL "Build C-Blosc2 benchmarks")
    set(BUILD_FUZZERS OFF CACHE BOOL "Build C-Blosc2 fuzzers")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    # We want the binaries of the C-Blosc2 library to go into the wheels
    set(BLOSC_INSTALL ON)
    add_subdirectory(c-blosc2)
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/c-blosc2/include")
    #target_link_libraries(blosc2_ext blosc2_static)
endif()

include_directories("${Python_NumPy_INCLUDE_DIRS}")

install(TARGETS blosc2_ext DESTINATION ${SKBUILD_PROJECT_NAME})
