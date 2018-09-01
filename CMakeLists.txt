cmake_minimum_required(VERSION 3.0)

if (NOT CMAKE_BUILD_TYPE)
        message(STATUS "No build type selected, default to Release")
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()
MESSAGE( STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE} )

project(rgbp)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

set (CMAKE_CXX_FLAGS_RELEASE "-Wall -Wcast-align -Ofast -DNDEBUG -std=c++17 -DHAVE_CXX0X -march=native")
set (CMAKE_CXX_FLAGS_DEBUG   "-Wall -Wcast-align -ggdb -std=c++17 -DHAVE_CXX0X -march=native")

add_library(rgbp INTERFACE)
add_library(rgbp::rgbp ALIAS rgbp)

target_include_directories(rgbp INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.42.0 COMPONENTS iostreams unit_test_framework filesystem system thread REQUIRED)
find_package(TBB REQUIRED)
find_package(GSL REQUIRED)

add_subdirectory(src)
