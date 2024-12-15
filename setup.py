import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        # Initialize without any sources; CMake handles them
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        # Ensure CMake is installed
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the C++ extensions.")

        # Build each extension
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        # Directory where the extension will be built
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        # Configure CMake arguments
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}"
        ]

        # Determine build configuration
        cfg = 'Debug' if self.debug else 'Release'
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
        build_temp = self.build_temp
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        print(ext.sourcedir)
        print(sys.executable)

        # Run CMake configuration
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp)

        # Run CMake build
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_temp)

setup(
    name='negative_binomial',
    version='0.1.0',
    author="Daniel O'Hanlon",
    author_email='dpohanlon@gmail.com',
    description='Python bindings for Negative Binomial PMF',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    ext_modules=[CMakeExtension('fast_negative_binomial')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
    packages=['fast_negative_binomial'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',
)
