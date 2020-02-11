/*
   nvcc
*/
#include <iostream>
#include "file_output.hpp"

int main(void) {
    file_output::test();
    int x = file_output::test2();
    std::cout << x;
    return 0;
}


void julia

/*
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

*/
