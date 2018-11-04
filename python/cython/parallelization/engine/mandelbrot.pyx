# distutils: extra_link_args = -fopenmp
# distutils: extra_compile_args = -fopenmp
import numpy as np

cimport numpy as cnp
cimport cython
from cython.parallel cimport prange


@cython.cdivision(True)
cdef float complex complex_from_pixel_loc(
        int pixel_x, int pixel_y,
        int pixel_width, int pixel_height,
        float max_x, float min_x, float max_y
) nogil:

    cdef float complex c = (
        min_x + (pixel_x / <float>(pixel_width))*(max_x - min_x)) + (
    -max_y + (pixel_y / <float>(pixel_height))*2*max_y)*1j
    return c

cdef int single_pixel_mandelbrot(float complex c, int max_iter, int max_value) nogil:
    cdef int k
    cdef float complex z = 0

    for k in range(max_iter):
        z = z*z + c
        if abs(z) > max_value:
            return k+1
    return 1

cdef int single_pixel_julia(float complex z, float complex c, int max_iter, int max_value) nogil:
    cdef int k

    for k in range(max_iter):
        z = z*z + c
        if abs(z) > max_value:
            return k+1
    return 1

@cython.boundscheck(False)
@cython.wraparound(False)
def mandelbrot(
        int width, int height,
        float img_max_x, float img_min_x, float img_max_y,
        int max_iter, int max_value
):
    cdef cnp.int32_t[:,:] arr = np.zeros((height, width), np.int32)
    cdef float complex c
    cdef int i, j

    for i in prange(0, height/2, schedule="guided", nogil=True):
        for j in range(width):
            c = complex_from_pixel_loc(j, i, width, height, img_max_x, img_min_x, img_max_y)
            arr[i, j] = single_pixel_mandelbrot(c, max_iter, max_value)
    # Mandelbrot set is symmetric about the x axis
    arr[height/2:] = arr[:height/2][::-1]

    return np.array(arr)


@cython.boundscheck(False)
@cython.wraparound(False)
def julia(
        int width, int height, float complex c,
        float img_max_x, float img_min_x, float img_max_y,
        int max_iter, int max_value
):
    cdef cnp.int32_t[:,:] arr = np.zeros((height, width), np.int32)
    cdef int i, j
    cdef float complex z0

    for i in prange(0, height, schedule="guided", nogil=True):
        for j in range(width):
            z0 = complex_from_pixel_loc(j, i, width, height, img_max_x, img_min_x, img_max_y)
            arr[i, j] = single_pixel_julia(z0, c, max_iter, max_value)

    return np.array(arr)
