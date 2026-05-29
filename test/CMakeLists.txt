cmake_minimum_required(VERSION 3.22)

project(espectre_tests LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ESPECTRE_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

include(CTest)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(Coverage)
include(TestMocks)
include(EspectreTestLib)

add_subdirectory(suites)
