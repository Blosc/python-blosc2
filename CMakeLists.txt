cmake_minimum_required(VERSION 3.15.0)
project(python-blosc2)
# Specifying Python version below is tricky, but if you don't specify the minimum version here,
# it would not consider python3 when looking for the executable. This is problematic since Fedora
# does not include a python symbolic link to python3.
#find_package(Python 3.10 COMPONENTS Interpreter NumPy Development.Module REQUIRED)
find_package(Python COMPONENTS Interpreter NumPy Development.Module REQUIRED)
target_link_libraries(blosc2_ext PRIVATE Python::Numpy)

add_subdirectory(blosc2)
