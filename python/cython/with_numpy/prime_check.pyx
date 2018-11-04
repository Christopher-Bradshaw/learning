import numpy as np

cimport numpy as cnp
cimport cython

from libc.math cimport ceil, sqrt

# There is a ton more yellow if we don't have these!
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def is_prime_numpy(input):
    cdef int i
    # There isn't great support of bool arrays so just cast from int
    cdef cnp.uint8_t[:] is_prime = np.zeros(len(input), dtype=np.uint8)

    # Note that in this array there is lots of yellow because we are using
    # python types. Even though they are numpy types.
    v = input[0]
    assert type(v) is np.int64
    assert type(v) is not long
    for i in range(len(input)):
        v = input[i]
        is_prime[i] = _is_prime(v)

    return np.array(is_prime, dtype=np.bool_)


# There is a ton more yellow if we don't have these!
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def is_prime_half_half(input):
    cdef int i
    cdef cnp.uint8_t[:] is_prime = np.zeros(len(input), dtype=np.uint8)
    cdef long v1 = 0
    assert type(v1) is not np.int64
    assert type(v1) is long

    # Pretty similar amount of yellow. We just move where we do the type conversion
    # a little earlier.
    for i in range(len(input)):
        v1 = input[i]
        is_prime[i] = _is_prime(v1)
    return np.array(is_prime, dtype=np.bool_)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def is_prime_memoryview(input):
    cdef int i
    cdef cnp.uint8_t[:] is_prime = np.zeros(len(input), dtype=np.uint8)
    cdef long v1 = 0
    assert type(v1) is not np.int64
    assert type(v1) is long


    # A lot less yellow in the loop as we do all the type conversion before
    # ascontiguousarray is needed to make sure that out input is in C array order.
    # c_input is a memoryview and has much fewer methods/attributes
    cdef long[:] c_input = np.ascontiguousarray(input)
    for i in range(len(c_input)):
        v1 = c_input[i]
        is_prime[i] = _is_prime(v1)
    return np.array(is_prime, dtype=np.bool_)


cdef cnp.uint8_t _is_prime(long n):
    cdef int i
    # sqrt casts to double, ceil keeps it as double, cast back to int
    cdef int max = int(ceil(sqrt(n+1)))

    for i in range(2, max):
        if n % i == 0:
            return False
    return True
