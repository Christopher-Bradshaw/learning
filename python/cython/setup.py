from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
        # exts to build - hence "build_ext" in the compile arg
        ext_modules = cythonize([
            'yield_and_function_overhead/funcs.pyx',
            'with_numpy/prime_check.pyx',
            'memory_views/memory_views.pyx',
            Extension('wrapping_c_code.wrapper', sources=['wrapping_c_code/wrapper.pyx', 'wrapping_c_code/c_funcs.c']),
        ]),
        # Where to find the *.h files needed to build things
        include_dirs = [
            np.get_include(),
        ],
)

# python setup.py build_ext --inplace
