cmake_minimum_required(VERSION 3.15...3.27)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C)

find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)

add_subdirectory(blosc2)




