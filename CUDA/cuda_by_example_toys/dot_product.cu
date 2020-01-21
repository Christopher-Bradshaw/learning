/*
   nvcc --run dot_product.cu helpers.c
*/

#include <stdio.h>
extern "C" {
#include "helpers.h"
}

#define N 1000 // The length of the vectors
#define threadsPerBlock 101

__global__ void dot(int *res, int *a, int *b) {
    // Each block will have its own cache. This is shared across
    // the threads in that block
    __shared__ int cache[threadsPerBlock];
    /* cudaMalloc(&dev_res, blockDim.x * sizeof(int)); */

    // The index is the blockID * the length of each block
    // plus the thread ID within that block.
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Each thread will track its sum
    int thread_sum = 0;
    // Each thread might need to hit multiple memory locations
    // Or it might hit none! In which case we write the 0 into the cache
    while (tid < N) {
        thread_sum += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    // Each thread now stores its result in the cache
    cache[threadIdx.x] = thread_sum;

    // We now need to sum the values in the cache. But before doing that
    // we need to make sure all threads have computed all their values.
    // We need a barrier.
    __syncthreads();

    // We now need to reduce the cache array using a sum operator.
    // Naively we could just have a single thread do this. But, better would
    // be to divide and conquer!
    // I'll be naive for now...
    if (threadIdx.x == 0) {
        for (int i=1; i<threadsPerBlock; i++) {
            cache[0] += cache[i];
        }
        // This is what is "returned"
        res[blockIdx.x] = cache[0];
    }
}

int main(void) {
    int blocks = 2;

    int res[blocks], a[N], b[N];
    arange(a, N);
    arange(b, N);

    // Allocate space on the GPU. Ignoring errors...
    int *dev_res, *dev_a, *dev_b;
    cudaMalloc(&dev_res, blocks * sizeof(int));
    cudaMalloc(&dev_a, N * sizeof(int));
    cudaMalloc(&dev_b, N * sizeof(int));

    // Copy data to the CPU
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);


    // Do the computation
    // We run 2 blocks and 10 threads
    dot<<<blocks, threadsPerBlock>>>(dev_res, dev_a, dev_b);

    // Move output array back to CPU
    cudaMemcpy(res, dev_res, blocks * sizeof(int), cudaMemcpyDeviceToHost);

    // Free the cuda arrays
    cudaFree(dev_res);
    cudaFree(dev_a);
    cudaFree(dev_b);

    // We sum up the results of the blocks on the CPU because reductions are pretty
    // inefficient on GPUs. Only the first few steps really use the parallelization.
    for (int i=1; i<blocks; i++) {
        res[0] += res[i];
    }
    printf("%d\n", res[0]);

    return 0;
}
