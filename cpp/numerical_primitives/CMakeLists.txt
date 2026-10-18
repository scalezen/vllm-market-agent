cmake_minimum_required(VERSION 3.20)
project(numprims_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Build against the first python on PATH (the active conda env locally,
# actions/setup-python on CI) and ask it where pybind11's CMake config is.
if(NOT DEFINED Python_EXECUTABLE)
  find_program(Python_EXECUTABLE NAMES python3 python REQUIRED)
endif()
if(NOT DEFINED pybind11_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
endif()
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core
    src/bindings.cpp
    src/topk_cosine.cpp
)

target_include_directories(_core PRIVATE include)

# Drop the built extension next to python/numerical_primitives/__init__.py so
# `from . import _core` in _backend.py finds it with no install step.
set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../python/numerical_primitives"
)
