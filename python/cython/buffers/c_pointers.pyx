import numpy as np
cimport numpy as cnp

cimport cython
# prefer the cpthon implementation - http://cython.readthedocs.io/en/latest/src/tutorial/memory_allocation.html
# from libc.stdlib cimport malloc, free
from cpython.mem cimport PyMem_Malloc, PyMem_Free


DEF STACK_SIZE = 1000000

@cython.boundscheck(False)
@cython.wraparound(False)
def arrange(int n):
    cdef int i
    cdef int *arranged = <int*>PyMem_Malloc(n * sizeof(int))

    if arranged == NULL:
        return -1

    for i in range(n):
        arranged[i] = i

    cdef int[::1] arranged_mv = <int[:n]>arranged
    return np.array(arranged_mv)


@cython.boundscheck(False)
@cython.wraparound(False)
def arrange10():
    cdef int i
    cdef int[10] arranged
    print(<unsigned long>&arranged[0])

    for i in range(10):
        arranged[i] = i

    return np.array(arranged)

# This assigns stuff onto the stack to overwrite what we previously aranged
def do_busy_work():
    cdef int[STACK_SIZE] stack_overwrite
    print(<unsigned long>&stack_overwrite[0])
    print(<unsigned long>&stack_overwrite[STACK_SIZE - 1])

    for i in range(STACK_SIZE):
        stack_overwrite[i] = -1

    return stack_overwrite




cdef class _memory_owner:
    cdef void *_data
    def __dealloc__(self):
        print("<---- Deep in the bowels of C ----> Freeing!")
        if self._data != NULL:
            PyMem_Free(self._data)

@cython.boundscheck(False)
@cython.wraparound(False)
def arrange_safe(int n):
    cdef int i
    cdef int *arranged = <int*>PyMem_Malloc(n * sizeof(int))
    print(<unsigned long>arranged)

    if arranged == NULL:
        return -1

    for i in range(n):
        arranged[i] = i



    # Create the numpy array as before
    cdef int[::1] arranged_mv = <int[:n]>arranged
    np_arr = np.array(arranged_mv)

    # Create a python object that owns the memory
    cdef _memory_owner mo = _memory_owner()
    mo._data = <void*>arranged

    # https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_SetBaseObject
    # This steals the ownership from mo! Now we own it so when we die (and there are no other views) we can call
    # the dealloc.
    cnp.set_array_base(np_arr, mo)
    return np_arr
