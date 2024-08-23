cmake_minimum_required(VERSION 3.15.0)
project(python-blosc2)
find_package(Python COMPONENTS Interpreter NumPy Development.Module REQUIRED)
include_directories("${Python_NumPy_INCLUDE_DIRS}")

add_subdirectory(blosc2)
