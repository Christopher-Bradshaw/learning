#include "julia.hpp"
#include <assert.h>
#include <complex>
#include <iostream>

__constant__ float c_dev[2];
__constant__ float edges[4];
__constant__ int n_pixels[2];
__constant__ float max_value_dev;

namespace julia {
// Declare this here because we don't want to export it
__global__ void julia_gpu_kernel(int *res);

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

void julia_gpu(int x_pixels, int y_pixels, cfloat c, float left_edge, float right_edge, float bottom_edge, float top_edge, int max_value, int *res) {
    // INSTRUMENT!
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    // Space on device
    int *dev_res;
    assert(cudaMalloc(&dev_res, x_pixels * y_pixels * sizeof(int)) == cudaSuccess);

    int n_blocks = 256;
    int n_threads = 32;

    float tmp_c[2] = {c.real(), c.imag()};
    float tmp_edges[4] = {left_edge, right_edge, bottom_edge, top_edge};
    int tmp_n_pixels[2] = {x_pixels, y_pixels};

    assert(cudaMemcpyToSymbol(c_dev, tmp_c, sizeof(tmp_c)) == cudaSuccess);
    assert(cudaMemcpyToSymbol(edges, tmp_edges, sizeof(tmp_edges)) == cudaSuccess);
    assert(cudaMemcpyToSymbol(max_value_dev, &max_value, sizeof(max_value)) == cudaSuccess);
    assert(cudaMemcpyToSymbol(n_pixels, tmp_n_pixels, sizeof(tmp_n_pixels)) == cudaSuccess);




    julia_gpu_kernel<<<n_blocks, n_threads>>>(dev_res);

    // Copy results back to the host
    assert(cudaMemcpy(res, dev_res, x_pixels * y_pixels * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);

    cudaEventRecord(stop);
    // ^ tells us to record an event when we get here. But we can't read the time off it until we've got there
    // So, we synchronize on that event.
    cudaEventSynchronize(stop);
    float t;
    cudaEventElapsedTime(&t, start, stop);
    std::cout << "Time taken: " << t << "ms" << std::endl;

}

__global__ void julia_gpu_kernel(int *res) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto width = edges[1] - edges[0];
    auto height = edges[3] - edges[2];

    int x, y;
    float pos_x, pos_y, pos_x_tmp;


    while (tid < n_pixels[0] * n_pixels[1]) {
        int count = 0;
        // This is real/imag in the normal version
        // The first part is the fraction along the image
        x = tid % n_pixels[0];
        y = tid / n_pixels[0];
        pos_x = (float)x / n_pixels[0] * width + edges[0];
        pos_y = (float)y / n_pixels[1] * height + edges[2];

        while(count < max_value_dev) {
            pos_x_tmp = std::pow(pos_x, 2) - std::pow(pos_y, 2) + c_dev[0];
            pos_y = 2 * pos_x * pos_y + c_dev[1];
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
