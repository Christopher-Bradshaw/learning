from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
        # exts to build - hence "build_ext" in the compile arg
        ext_modules = cythonize([
            'yield_and_function_overhead/funcs.pyx',
            'with_numpy/prime_check.pyx',
        ]),
        # Where to find the *.h files needed to build things
        include_dirs = [
            np.get_include(),
        ],
)

# python setup.py build_ext --inplace
