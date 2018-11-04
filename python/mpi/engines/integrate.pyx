cimport cython
from libc.math cimport sin

import numpy as np

cdef double squared(double x):
    return x*x

@cython.cdivision(True)
def integrate(double start, double stop, double num_pts, str func_string):
    cdef double running_sum = 0
    cdef double step = (stop - start)/num_pts
    cdef double x = start + step/2

    cdef double (*f) (double)

    if func_string == "squared":
        f = squared
    elif func_string == "sin":
        f = sin
    else:
        raise ValueError("Unrecognized function string!")

    while x < stop:
        running_sum += f(x)
        x += step
    return (stop - start) * running_sum / num_pts
