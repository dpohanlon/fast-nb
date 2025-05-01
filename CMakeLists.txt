# CMakeLists.txt for building and linking Google Benchmark, Pybind11, and Boost
cmake_minimum_required(VERSION 3.10...3.18)
project(FastNB LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to enable Benchmark (OFF by default)
option(ENABLE_BENCHMARK "Enable Google Benchmark" OFF)

# Option to enable Boost (ON by default)
option(ENABLE_BOOST "Enable Boost Libraries" ON)

set(PROJECT_VERSION 0.1)

# Allow specifying a custom Clang compiler path
if(DEFINED CLANG_PATH)
    set(CMAKE_CXX_COMPILER ${CLANG_PATH})
endif()

# Find required packages
if(ENABLE_BENCHMARK)
    find_package(benchmark REQUIRED)
endif()

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
#find_package(pybind11 CONFIG REQUIRED)
#add_subdirectory(extern/pybind11)
if(NOT TARGET pybind11::module)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11)
endif()

# Find Threads
find_package(Threads REQUIRED)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
#include_directories( "/usr/local/include/eigen3" )

# Find Boost
if(ENABLE_BOOST)
    # Specify the Boost components you need
    set(BOOST_COMPONENTS filesystem system sort)

    # Optionally specify the Boost root if it's in a non-standard location
    # set(BOOST_ROOT "/path/to/boost")
    # set(BOOST_INCLUDEDIR "/path/to/boost/include")
    # set(BOOST_LIBRARYDIR "/path/to/boost/lib")

    set(Boost_USE_STATIC_LIBS ON)
    find_package(Boost 1.70.0 REQUIRED system)

    if(Boost_FOUND)
        message(STATUS "Boost version: ${Boost_VERSION}")
        message(STATUS "Boost include dirs: ${Boost_INCLUDE_DIRS}")
        message(STATUS "Boost libraries: ${Boost_LIBRARIES}")
    else()
        message(FATAL_ERROR "Boost not found!")
    endif()
endif()

# Create a library for the core functionality
add_library(fast_nb_lib STATIC fast_negative_binomial/fast_nb.cpp)
add_library(FastNB::fast_nb_lib ALIAS fast_nb_lib)

# Specify include directories for the library
#target_include_directories(fast_nb_lib PUBLIC fast_negative_binomial)

# Add executable target
add_executable(fast_nb fast_negative_binomial/main.cpp)

#target_include_directories(fast_nb_lib PUBLIC
#    fast_negative_binomial
#    ${Boost_INCLUDE_DIRS}  # Ensure Boost includes are PUBLIC
#)

target_include_directories(fast_nb_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${Boost_INCLUDE_DIRS}
)

# Link the executable with the core library and Threads
target_link_libraries(fast_nb PRIVATE fast_nb_lib Threads::Threads)

# Link Benchmark only if enabled
if(ENABLE_BENCHMARK)
    target_link_libraries(fast_nb PRIVATE benchmark::benchmark)
    target_compile_definitions(fast_nb PRIVATE ENABLE_BENCHMARK=1)
endif()

# Link Eigen
target_link_libraries(fast_nb PUBLIC Eigen3::Eigen)
target_link_libraries(fast_nb_lib PUBLIC Eigen3::Eigen)

target_link_libraries(fast_nb PRIVATE Boost::headers)

# Create the Python module using Pybind11
pybind11_add_module(fast_negative_binomial fast_negative_binomial/bindings.cpp)

# Link the Python module with the core library
target_link_libraries(fast_negative_binomial PRIVATE fast_nb_lib)

# Set compiler flags based on the environment - maybe we can be a bit less strict
# for Apple silicon?
if(DEFINED ENV{GITHUB_ACTIONS} AND "$ENV{GITHUB_ACTIONS}" STREQUAL "true")
    message(STATUS "Building on GitHub Actions: Using no arch specific flags")
    target_compile_options(fast_nb_lib PRIVATE -O3 -ffast-math)
    target_compile_options(fast_nb PRIVATE -O3 -ffast-math)
else()
    message(STATUS "Building locally: Using -march=native")
    target_compile_options(fast_nb_lib PRIVATE -O3 -march=native -ffast-math)
    target_compile_options(fast_nb PRIVATE -O3 -march=native -ffast-math)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fast_nb_lib PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(fast_nb PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(fast_negative_binomial PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS fast_negative_binomial
        LIBRARY DESTINATION fast_negative_binomial)

# Install the static library
install(TARGETS fast_nb_lib
    EXPORT FastNBTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Install the executable
install(TARGETS fast_nb
    RUNTIME DESTINATION bin
)

# Install the Python module
install(TARGETS fast_negative_binomial
    LIBRARY DESTINATION lib/python_modules
)

# Install header files
install(DIRECTORY fast_negative_binomial/
    DESTINATION include/fast_negative_binomial
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

# Export the targets to a CMake script
install(EXPORT FastNBTargets
    FILE FastNBTargets.cmake
    NAMESPACE FastNB::
    DESTINATION lib/cmake/FastNB
)

# Create a package configuration file
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/FastNBConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FastNBConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/FastNBConfig.cmake"
    INSTALL_DESTINATION lib/cmake/FastNB
)

# Install the package configuration files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/FastNBConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/FastNBConfigVersion.cmake"
    DESTINATION lib/cmake/FastNB
)

# Optionally, install CMake scripts (if any)
# install(DIRECTORY cmake/ DESTINATION lib/cmake/FastNB)
