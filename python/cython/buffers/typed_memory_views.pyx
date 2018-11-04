cimport cython
cimport numpy as cnp
import numpy as np
from libc.math cimport HUGE_VAL

# Needs to be called with some object that implements the buffer protocol and has
# type byte. mv is then a view into the buffer of that object and we can quickly
# iterate over it.
@cython.boundscheck(False)
@cython.wraparound(False)
def summer(char[:] mv):
    cdef int i, s = 0

    # Note that we definitely *should not* do
    # for c in mv:
    # As that hits many python APIs (PyGetItem)
    for i in range(len(mv)):
        s += mv[i] # Without boundscheck/wraparound this is much faster
    return s

@cython.boundscheck(False)
@cython.wraparound(False)
def mean(double[:, ::1] mv): # Note that specifying C/F contigousness is the only "slicing" you can do in the func def
    cdef int i, j
    cdef double s = 0

    for i in range(mv.shape[0]):
        for j in range(mv.shape[1]):
            s += mv[i,j]
            # Note that in multiple loops, you should make sure the index of the inner loop
            # is the one with the smallest stride. This way you get cache hits.
            # Reversing the order of these for loops slows this down by 8x

    return s/(mv.shape[0] * mv.shape[1])


ctypedef fused most_numerics:
    cython.char
    cython.float
    cython.double
    cython.int
    cython.long

@cython.boundscheck(False)
@cython.wraparound(False)
def generic_mean(most_numerics[:] mv):
    cdef int i, j
    cdef double s = 0

    for i in range(len(mv)):
        s += mv[i]

    return s/(len(mv))


@cython.boundscheck(False)
@cython.wraparound(False)
def minimum_with_slicing(arr):
    cdef int i, j
    cdef double cur_min = HUGE_VAL

    cdef double[:,:] mv = arr[1:10:3, :50:2]
    print(dir(mv))
    assert mv.shape[0] == 3 and mv.shape[1] == 25 # from how we sliced it on assignment
    assert mv.strides[0] == 3*8*1000 # originally this is a 1000 x 1000 matrix. x8 because it is a double. x3 because of the step
    assert mv.strides[1] == 16 # step on the second dim of 2x8
    assert mv.is_c_contig() == False and mv.is_f_contig() == False # neither because of the slicing
    assert mv.ndim == 2


    # We can also add dimensions easily, as in numpy

    cdef double[:,:,:] mv2 = mv[:, None, :]
    print(mv2.shape)

    for i in range(mv.shape[0]):
        for j in range(mv.shape[1]):
            if mv[i,j] < cur_min:
                cur_min = mv[i,j]

    return cur_min


# To work with a structured array, you need to define the structure.
# This ... actually makes a lot of sense
ctypedef struct arr_dtype:
    cnp.int32_t a
    cnp.float32_t b

# Then you cast your structured array to a memoryview with the type of that structure
@cython.boundscheck(False)
@cython.wraparound(False)
def doubler_structured_memory_views(arr_dtype[:] mv):
    # Accessing individual rows/element
    print(mv[0])
    print(mv[0].a)
    print(mv[0].b)

    # You can't do these numpy like things
    # print(mv[0]["b"])
    # print(mv.b) -> No nice col slicing. You could probably hack this

    cdef int i

    for i in range(len(mv)):
        mv[i].a *= 2
        mv[i].b *= 2

    return np.array(mv, dtype=[("a", np.int32), ("b", np.float32)])
