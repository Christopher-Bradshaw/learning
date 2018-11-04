# cython: profile=True
from libc.math cimport sin

def sin_squared(double x):
    return sin(x)**2

def integrate(double start, double stop, func, int n=200000):
    cdef double dx, total
    cdef int i

    dx = (stop - start) / n
    total = 0
    for i in range(n):
        total += func(start + (i+0.5)*dx)
    return total * dx

cdef double c_sin_squared(double x):
    return sin(x)**2

def integrate_sin_squared(double start, double stop, int n=200000):
    return c_integrate(start, stop, c_sin_squared, n)



# Ideally we would be able to do this, but because we allow python to set the function
# We can't type it fully. Compare the yellow to integrate to see the difference.
cdef double c_integrate(
        double start,
        double stop,
        double (*func)(double),
        int n=200000,
):
    cdef double dx, total
    cdef int i

    dx = (stop - start) / n
    total = 0
    for i in range(n):
        total += func(start + (i+0.5)*dx)
    return total * dx
