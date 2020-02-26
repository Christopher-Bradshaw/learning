#include "julia.hpp"
#include <complex>
#include <iostream>

namespace julia {
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
                pos = julia::iter_julia(pos, c);
                if (std::abs(pos) >= 2) {
                    break;
                }
                res[x][y] += 1;
            }
        }
    }
}

cfloat iter_julia(cfloat z_old, cfloat c) { return std::pow(z_old, 2) + c; }
}  // namespace julia
