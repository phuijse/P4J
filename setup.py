import os
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

include_dirs = ['.', np.get_include()]
library_dirs = []
if os.name == 'nt':  # Windows, assumming MSVC compiler
    libraries = []
    compiler_args = ['/Ox', '/fp:fast']
elif os.name == 'posix':  # UNIX, assumming GCC compiler
    libraries = ['m']
    compiler_args = ['-O3', '-ffast-math']

extensions = [
    Extension("*",
              sources=[os.path.join("src", "P4J", "algorithms", "*.pyx")],
              extra_compile_args=compiler_args,
              include_dirs=include_dirs,
              libraries=libraries,
              library_dirs=library_dirs
              )]

setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=cythonize(extensions,
                          annotate=False,
                          language_level=3),
)
