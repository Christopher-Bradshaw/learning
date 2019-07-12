# C as python accelerator

Rather than implementing things in cython, we can implement in C. There are then a variety of ways to use that code in python. The easiest (imo) is to use cython to wrap the C code.

## Steps

* Clean up results from previous compilations `make clean`
* Build the extension using the `python3 setup.py build_ext --inplace`
* Run the code `python3 main.py`

## What just happened?

We have some C library (`c_funcs`).
We then implement some functions in our cython wrapper. These functions are just thin wrappers around the C functions (though in practice we might want to change the API here). However, we need to define these functions so that cython handles the python-C interop.
We include the `.h` file in the `wrapper.pyx`. This is necessary because this is compiled to `wrapper.c` and to compile this `.c` to an object file we need to know function signatures!
Finally, we tell the build system (`setup.py`) that the extension consists of both the wrapper and the funcs.

## Other options

From doing this with ctypes in the rust example, I now think that that might be even easier...
