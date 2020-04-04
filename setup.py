from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

setup(
    ext_modules = cythonize(
        Extension(
            '_forest',
            sources = ['doubt/estimators/_forest.pyx'],
            include_dirs = [np.get_include()]
        )
    ),
    install_requires = ['numpy']
)

os.system('mv *.so doubt/estimators/')
