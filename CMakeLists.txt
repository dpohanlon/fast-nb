# CMakeLists.txt for building and linking Google Benchmark
cmake_minimum_required(VERSION 3.10)
project(FastNB LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Allow specifying a custom Clang compiler path
if(DEFINED CLANG_PATH)
    set(CMAKE_CXX_COMPILER ${CLANG_PATH})
endif()

# Find Google Benchmark installed via Homebrew
find_package(benchmark REQUIRED)

# Add executable target
add_executable(fast_nb fast_nb.cpp)

# Link Google Benchmark and pthread
find_package(Threads REQUIRED)
target_link_libraries(fast_nb
    benchmark::benchmark
    Threads::Threads
)

# Set optimization flags for performance
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(fast_nb PRIVATE -O3 -march=native -ffast-math)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  target_compile_options(fast_nb PRIVATE -O3 -march=native -ffast-math)
endif()

# Ensure OpenMP is not enforced on Apple Silicon
if(NOT APPLE)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(fast_nb OpenMP::OpenMP_CXX)
    endif()
endif()
