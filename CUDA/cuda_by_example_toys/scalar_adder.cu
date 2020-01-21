#include <stdio.h>

__global__ void add(int *res, int a, int b) {
    *res = a + b;
}

int main(void) {
    int res;
    int *device_res = NULL;

    cudaError_t mres;
    // Allocate memory on the device
    // You cannot dereference this in host code!
    mres = cudaMalloc(&device_res, sizeof(int));
    if (mres != cudaSuccess) {
        printf("Malloc failed\n");
        return -1;
    }
    // Do computation
    add<<<1,1>>>(device_res, 2, 7);
    // Copy result back to host
    mres = cudaMemcpy(&res, device_res, sizeof(int), cudaMemcpyDeviceToHost);
    if (mres != cudaSuccess) {
        printf("Memcpy failed\n");
        return -1;
    }

    // Now free the memory we allocated on the device
    mres = cudaFree(device_res);
    if (mres != cudaSuccess) {
        printf("Free failed\n");
        return -1;
    }

    printf("2 + 7 = %d\n", res);
    return 0;
}

