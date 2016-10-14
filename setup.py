import re
import io
import os
from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()

def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def version(path):
    """
    https://packaging.python.org/en/latest/single_source_version.html
    """
    version_file = read(path)
    version_match = re.search(r"""^__version__ = ['"]([^'"]*)['"]""",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")
    
setup(
    name = 'P4J',
    packages = ['P4J'], 
    version = version('P4J/__init__.py'),
    description = 'Periodic light curve analysis tools based on Information Theory',
    long_description=readme(),
    author = 'Pablo Huijse',
    author_email = 'pablo.huijse@gmail.com',
    license='MIT',
    url = 'https://github.com/phuijse/P4J', 
    download_url = 'https://github.com/phuijse/P4J/tarball/stable', 
    keywords = ['astronomy periodic time series correntropy'], 
    install_requires=[
        'numpy',
        'scipy'
    ],
    classifiers = [
        'Natural Language :: English',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: MIT License',
        'Environment :: Console',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
