# The point of this is to test the speed of generators/function calls in cython

def counter(int x):
    cdef long count = 0
    cdef int i
    for i in range(x):
        count += i
    return count


def counter_that_calls_a_func(int x):
    cdef long count = 0
    cdef int i
    for i in range(x):
        count += _count_helper(i)
    return count

cdef int _count_helper(int i):
    return i

def counter_that_calls_a_generator(int x):
    cdef long count = 0
    cdef int i
    for i in _count_helper_generator(x):
        count += i
    return count

# yield can't be used in a cdef'ed func because C doesn't have closures
def _count_helper_generator(int x):
    cdef int i
    for i in range(x):
        yield i
