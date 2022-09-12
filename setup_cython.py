from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("src/GCMProject/state_evolution/auxiliary/cython_functions.pyx"),
    include_dirs=[numpy.get_include()]
)
