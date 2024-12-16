# CMakeLists.txt for building and linking Google Benchmark and Pybind11
cmake_minimum_required(VERSION 3.4...3.18)
project(FastNB LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to enable Benchmark (OFF by default)
option(ENABLE_BENCHMARK "Enable Google Benchmark" OFF)

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

# Create a library for the core functionality
add_library(fast_nb_lib STATIC fast_negative_binomial/fast_nb.cpp)

# Specify include directories for the library
target_include_directories(fast_nb_lib PUBLIC fast_negative_binomial)

# Add executable target
add_executable(fast_nb fast_negative_binomial/main.cpp)

# Link the executable with the core library and Threads
target_link_libraries(fast_nb PRIVATE fast_nb_lib Threads::Threads)

# Link Benchmark only if enabled
if(ENABLE_BENCHMARK)
    target_link_libraries(fast_nb PRIVATE benchmark::benchmark)
    target_compile_definitions(fast_nb PRIVATE ENABLE_BENCHMARK=1)
endif()

# Create the Python module using Pybind11
pybind11_add_module(fast_negative_binomial fast_negative_binomial/bindings.cpp)

# Link the Python module with the core library
target_link_libraries(fast_negative_binomial PRIVATE fast_nb_lib)

# Set optimization flags for performance
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(fast_nb_lib PRIVATE -O3 -march=native -ffast-math)
    target_compile_options(fast_nb PRIVATE -O3 -march=native -ffast-math)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(fast_nb_lib PRIVATE -O3 -march=native -ffast-math)
    target_compile_options(fast_nb PRIVATE -O3 -march=native -ffast-math)
endif()

# Ensure OpenMP is not enforced on Apple Silicon
if(NOT APPLE)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(fast_nb_lib PRIVATE OpenMP::OpenMP_CXX)
        target_link_libraries(fast_nb PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

install(TARGETS fast_negative_binomial
        LIBRARY DESTINATION fast_negative_binomial)
