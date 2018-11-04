import numpy as np
import time

cimport cython
from libc.stdlib cimport rand, RAND_MAX, srand

@cython.cdivision(True)
def estimate_pi(int count):
    cdef int count_in_circle = 0
    cdef float x, y

    srand(time.time())

    for _ in range(count):
        x = float(rand()) / RAND_MAX
        y = float(rand()) / RAND_MAX
        if x**2 + y**2 < 1:
            count_in_circle += 1
    return 4*float(count_in_circle)/count
