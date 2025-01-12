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
add_subdirectory(extern/pybind11)

# Find Threads
find_package(Threads REQUIRED)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Find Boost
if(ENABLE_BOOST)
    # Specify the Boost components you need
    set(BOOST_COMPONENTS filesystem system math)  # Added 'math'

    # Optionally specify the Boost root if it's in a non-standard location
    # set(BOOST_ROOT "/path/to/boost")
    # set(BOOST_INCLUDEDIR "/path/to/boost/include")
    # set(BOOST_LIBRARYDIR "/path/to/boost/lib")

    find_package(Boost 1.70.0 REQUIRED system math)

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

# Specify include directories for the library
target_include_directories(fast_nb_lib PUBLIC fast_negative_binomial)

# Add executable target
add_executable(fast_nb fast_negative_binomial/main.cpp)

target_include_directories(fast_nb_lib PUBLIC
    fast_negative_binomial
    ${Boost_INCLUDE_DIRS}  # Ensure Boost includes are PUBLIC
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

# Link Boost libraries if enabled
#if(ENABLE_BOOST)
# Include Boost headers
#target_include_directories(fast_nb_lib PRIVATE ${Boost_INCLUDE_DIRS})
#target_include_directories(fast_nb PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(fast_nb PRIVATE Boost::headers Boost::system)

# If you're using Boost in the Python module as well
# target_link_libraries(fast_negative_binomial PRIVATE fast_nb_lib Boost::filesystem Boost::system)
#endif()

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
