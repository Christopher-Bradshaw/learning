# Cython

See https://github.com/aphearin/accelerating_python to get started.

Cython: A Guide for Python Programmers (CAGFPP) is a good reference.

## Sections
* `compiling`: How to get your cython code useable. [CAGFPP chaper 2]
* `wrapping_c_code`: How to write cython that wraps C libs and makes it accessible to python code. [CAGFPP chapter 7]
* `profiling`: How to profile your python + cython. [CAGFPP chapter 9]
* `buffers`: How to work with numpy arrays (and the like). [CAGFPP chapter 10]
* `parallelization`: Multithreading in cython. [CAGFPP chapter 12]
* `with_numpy`: A concrete example of speed-ups when passing around numpy arrays. I did this before digging into buffers so at a much higher level.
* `yield_and_function_overhead`: A simple cython example. Tests function call overhead and was where I learned that you can't yield from a `cdef`ed func.


## Imports
* `cimport` pulls from https://github.com/cython/cython/tree/master/Cython/Includes
* `import` is the same as python
