#include "julia.hpp"
#include <assert.h>
#include <complex>
#include <iostream>

__constant__ float c_dev[2];
__constant__ float edges[4];
__constant__ float size[2];
__constant__ int n_pixels[2];
__constant__ int max_value_dev;

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

    float tmp_c[2] = {c.real(), c.imag()};
    float tmp_edges[4] = {left_edge, right_edge, bottom_edge, top_edge};
    float tmp_size[2] = {right_edge - left_edge, top_edge - bottom_edge};
    int tmp_n_pixels[2] = {x_pixels, y_pixels};

    assert(cudaMemcpyToSymbol(c_dev, tmp_c, sizeof(tmp_c)) == cudaSuccess);
    assert(cudaMemcpyToSymbol(edges, tmp_edges, sizeof(tmp_edges)) == cudaSuccess);
    assert(cudaMemcpyToSymbol(max_value_dev, &max_value, sizeof(max_value)) == cudaSuccess);
    assert(cudaMemcpyToSymbol(n_pixels, tmp_n_pixels, sizeof(tmp_n_pixels)) == cudaSuccess);
    assert(cudaMemcpyToSymbol(size, tmp_size, sizeof(tmp_size)) == cudaSuccess);

    // When working in more than 1d we don't do the loop. We just spin up enough blocks to cover the image.
    dim3 threadsPerBlock(16, 16);
    dim3 blocks(x_pixels / 16, y_pixels / 16);

    julia_gpu_kernel<<<blocks, threadsPerBlock>>>(dev_res);

    // Copy results back to the host
    assert(cudaMemcpy(res, dev_res, x_pixels * y_pixels * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaFree(dev_res) == cudaSuccess);

    cudaEventRecord(stop);
    // ^ tells us to record an event when we get here. But we can't read the time off it until we've got there
    // So, we synchronize on that event.
    cudaEventSynchronize(stop);
    float t;
    cudaEventElapsedTime(&t, start, stop);
    std::cout << "Time taken: " << t << "ms" << std::endl;

}

__global__ void julia_gpu_kernel(int *res) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * n_pixels[0];

    float pos_x, pos_y, pos_x_tmp;


    if ((x < n_pixels[0]) && (y <= n_pixels[1])) {
        int count = 0;
        // This is real/imag in the normal version
        // The first part is the fraction along the image
        pos_x = (float)x / n_pixels[0] * size[0] + edges[0];
        pos_y = (float)y / n_pixels[1] * size[1] + edges[2];

        while(count < max_value_dev) {
            pos_x_tmp = pos_x * pos_x - pos_y * pos_y + c_dev[0];
            pos_y = 2 * pos_x * pos_y + c_dev[1];
            pos_x = pos_x_tmp;
            if (pos_x * pos_x + pos_y * pos_y >= 4) {
                break;
            }
            count++;
        }
        res[offset] = count;
    }
}

}  // namespace julia
