#include <complex>
#include <iostream>
#include "file_output.hpp"

typedef std::complex<float> cfloat;

void julia(int, int, cfloat, float, float, float, float, int, int **);
cfloat iter_julia(cfloat, cfloat);

int main(void) {
    cfloat c = {0.5, 0.4};
    int x_pixels = 2000, y_pixels = 2000;
    int max_value = 200;
    // Allocate the result array
    int **res = (int **)calloc(x_pixels, sizeof(int *));
    for (int i = 0; i < x_pixels; i++) {
        res[i] = (int *)calloc(y_pixels, sizeof(int));
    }

    julia(x_pixels, y_pixels, c, -1.5, 1.5, -1.5, 1.5, max_value, res);

    file_output::write_ppm(res, x_pixels, y_pixels, max_value);
    file_output::write_jpg(res, x_pixels, y_pixels, max_value);
}

void julia(int x_pixels, int y_pixels, cfloat c, float left_edge,
           float right_edge, float bottom_edge, float top_edge, int max_value,
           int **res) {
    auto width = right_edge - left_edge;
    auto height = top_edge - bottom_edge;

    for (int x = 0; x < x_pixels; x++) {
        for (int y = 0; y < y_pixels; y++) {
            // This is not quite right - at bottom left of pixel
            cfloat pos = {(float)x / x_pixels * width + left_edge,
                          (float)y / y_pixels * height + bottom_edge};

            while (res[x][y] < max_value) {
                pos = iter_julia(pos, c);
                if (std::abs(pos) >= 2) {
                    break;
                }
                res[x][y] += 1;
            }
        }
    }
}

cfloat iter_julia(cfloat z_old, cfloat c) { return std::pow(z_old, 2) + c; }
