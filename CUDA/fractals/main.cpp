#include <complex>
#include <iostream>
#include "common.hpp"
#include "file_output.hpp"
#include "julia.hpp"

int main(void) {
    cfloat c = {0.5, 0.4};
    int x_pixels = 4096, y_pixels = 4096;
    /* int x_pixels = 1024, y_pixels = 1024; */
    int max_value = 200;
    // Allocate the result array
    int *res = (int *)calloc(x_pixels * y_pixels, sizeof(int));

    julia::julia_gpu(x_pixels, y_pixels, c, -1.5, 1.5, -1.5, 1.5, max_value,
                     res);

    /* file_output::write_ppm(res, x_pixels, y_pixels, max_value); */
    /* file_output::write_jpg(res, x_pixels, y_pixels, max_value); */
}
