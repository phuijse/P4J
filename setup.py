import re
import io
import os
from setuptools import setup
from setuptools.extension import Extension

try:
    import numpy as np
except ImportError:
    raise ImportError("Please install Numpy before installing P4J")


include_dirs = ['.', np.get_include()]
library_dirs = []
if os.name == 'nt':  # Windows, assuming MSVC compiler
    libraries = []
    compiler_args = ['/Ox', '/fp:fast']
elif os.name == 'posix':  # UNIX, assuming GCC compiler
    libraries = ['m']
    compiler_args = ['-O3', '-ffast-math', '-march=native', '-mtune=native', '-flto']
else:
    raise Exception('Unsupported operating system')


extensions = []
for file in os.listdir(os.path.join("P4J", "algorithms")):
    if file.endswith('.pyx'):
        extensions.append(
            Extension(
                '.'.join(['P4J', 'algorithms', file.rstrip('.pyx')]),
                sources=[os.path.join("P4J", "algorithms", file)],
                extra_compile_args=compiler_args,
                include_dirs=include_dirs,
                libraries=libraries,
                library_dirs=library_dirs
            )
        )

"""
Allow users to install the module even if they do not have cython.
If cython is not found the c sources are compiled instead. More details at:
https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules
"""
try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    import warnings
    USE_CYTHON = False
    warnings.warn('Cython not found, compiling from c sources')


def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in ('.pyx', '.py'):
                if extension.language == 'c++':
                    ext = '.cpp'
                else:
                    ext = '.c'
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions


if USE_CYTHON:
    extensions = cythonize(
        extensions,
        annotate=True,
        compiler_directives={'language_level': "3"},
        force=True,
        nthreads=4)
else:
    extensions = no_cythonize(extensions)


"""
    Read version automatically from __init__.py
    https://packaging.python.org/en/latest/single_source_version.html
"""


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def version(path):
    version_file = read(path)
    version_match = re.search(r"""^__version__ = ['"]([^'"]*)['"]""",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


"""
Actual setup
"""
setup(
    name='P4J',
    packages=['P4J'],
    ext_modules=extensions,
    version=version('P4J/__init__.py'),
    description='Periodic light curve analysis tools based on Information Theory',
    long_description=open('README.rst').read(),
    author='Pablo Huijse',
    author_email='pablo.huijse@gmail.com',
    license='MIT',
    url='https://github.com/phuijse/P4J',
    keywords=['astronomy periodic time series correntropy'],
    install_requires=[
        'numpy >=1.9.0',
    ],
    classifiers=[
        'Natural Language :: English',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: C',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
