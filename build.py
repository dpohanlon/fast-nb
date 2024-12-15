import os
import sys
import subprocess

import tempfile

# Well this sucks

def build_extension(ext):

    try:
        subprocess.check_output(['cmake', '--version'])
    except OSError:
        raise RuntimeError("CMake must be installed to build the C++ extensions.")

    # Directory where the extension will be built
    extdir = os.path.abspath(ext)
    if not extdir.endswith(os.path.sep):
        extdir += os.path.sep

    # Configure CMake arguments
    cmake_args = [
        f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
        f"-DPYTHON_EXECUTABLE={sys.executable}"
    ]

    cfg = 'Release'
    build_args = ['--config', cfg]

    # Platform-specific settings
    if sys.platform.startswith('win'):
        cmake_args += [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
        ]
        if sys.maxsize > 2**32:
            cmake_args += ["-A", "x64"]
        build_args += ["--", "/m"]
    else:
        cmake_args += [f"-DCMAKE_BUILD_TYPE={cfg}"]
        build_args += ["--", "-j2"]

    # Create build directory
    with tempfile.TemporaryDirectory() as build_temp:
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        sourcedir = os.path.abspath('')

        # Run CMake configuration
        subprocess.check_call(['cmake', sourcedir] + cmake_args, cwd=build_temp)

        # Run CMake build
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_temp)

build_extension("fast_negative_binomial")
