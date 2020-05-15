from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

setup(
    ext_modules = cythonize(
        Extension(
            'data_structures',
            sources = ['./data_structures.pyx'],
            #include_dirs = [np.get_include()]
        ),
        language_level = 3
    ),
    #install_requires = ['numpy']
)
