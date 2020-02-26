#include "julia.hpp"
#include <complex>
#include <iostream>

namespace julia {
// Declare this here because we don't want to export it
__global__ void julia_gpu_kernel(int x_pixels, int y_pixels, float c_real, float c_imag, float left_edge, float right_edge, float bottom_edge, float top_edge, int max_value, int *res);

void julia(int x_pixels, int y_pixels, cfloat c, float left_edge,
           float right_edge, float bottom_edge, float top_edge, int max_value,
           int *res) {
    auto width = right_edge - left_edge;
    auto height = top_edge - bottom_edge;

    for (int x = 0; x < x_pixels; x++) {
        for (int y = 0; y < y_pixels; y++) {
            // This is not quite right - at bottom left of pixel
            cfloat pos = {(float)x / x_pixels * width + left_edge,
                          (float)y / y_pixels * height + bottom_edge};

            while (res[x + y * x_pixels] < max_value) {
                pos = julia::iter_julia(pos, c);
                if (std::abs(pos) >= 2) {
                    break;
                }
                res[x+y * x_pixels] += 1;
            }
        }
    }
}
cfloat iter_julia(cfloat z_old, cfloat c) { return std::pow(z_old, 2) + c; }

void julia_gpu(int x_pixels, int y_pixels, cfloat c, float left_edge, float right_edge, float bottom_edge, float top_edge, int max_value,
           int *res) {
    // Space on device
    int *dev_res;
    cudaMalloc(&dev_res, x_pixels * y_pixels * sizeof(int));

    int n_blocks = 256;
    int n_threads = 32;
    julia_gpu_kernel<<<n_blocks, n_threads>>>(x_pixels, y_pixels, c.real(), c.imag(), left_edge, right_edge, bottom_edge, top_edge, max_value, dev_res);

    // Copy results back to the host
    cudaMemcpy(res, dev_res, x_pixels * y_pixels * sizeof(int), cudaMemcpyDeviceToHost);

}

__global__ void julia_gpu_kernel(int x_pixels, int y_pixels, float c_real, float c_imag, float left_edge, float right_edge, float bottom_edge, float top_edge, int max_value, int *res) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto width = right_edge - left_edge;
    auto height = top_edge - bottom_edge;

    int x, y;
    float pos_x, pos_y, pos_x_tmp;


    while (tid < x_pixels * y_pixels) {
        int count = 0;
        // This is real/imag in the normal version
        // The first part is the fraction along the image
        x = tid % x_pixels;
        y = tid / x_pixels;
        pos_x = (float)x / x_pixels * width + left_edge;
        pos_y = (float)y / y_pixels * height + bottom_edge;

        while(count < max_value) {
            pos_x_tmp = std::pow(pos_x, 2) - std::pow(pos_y, 2) + c_real;
            pos_y = 2 * pos_x * pos_y + c_imag;
            pos_x = pos_x_tmp;
            if (std::pow(pos_x, 2) + std::pow(pos_y, 2) >= 4) {
                break;
            }
            count++;
        }
        res[tid] = count;
        tid += blockDim.x * gridDim.x;
    }
}

}  // namespace julia
