from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name = 'P4J',
    packages = ['P4J'], 
    version = '0.3',
    description = 'Periodic light curve analysis tools based on Information Theory',
    long_description=readme(),
    author = 'Pablo Huijse',
    author_email = 'pablo.huijse@gmail.com',
    license='MIT',
    url = 'https://github.com/phuijse/P4J', 
    download_url = 'https://github.com/phuijse/P4J/tarball/0.1', 
    keywords = ['astronomy periodic time series correntropy'], 
    install_requires=[
        'numpy'
    ],
    classifiers = [
        'Natural Language :: English',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: MIT License',
        'Environment :: Console',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
