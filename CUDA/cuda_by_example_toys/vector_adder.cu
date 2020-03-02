#include <stdio.h>

#define N 100

__global__ void add_block(int *res, int *a, int *b) {
    // blockIdx.x tells us which item this kernel is
    // blockIdx is a built in variable that the CUDA runtime defines.
    int bid = blockIdx.x;
    // We can even print it in here! No guarantee of order though
    printf("%d\n", bid);
    res[bid] = a[bid] + b[bid];
}

__global__ void add_thread(int *res, int *a, int *b) {
    int tid = threadIdx.x;
    res[tid] = a[tid] + b[tid];
}


int main(void) {
    int res[N], a[N], b[N];

    // Populate the arrays with something
    for (int i=0; i<N; i++) {
        a[i] = i;
        b[i] = 2*i;
    }

    // Allocate space on the GPU. Ignoring errors...
    int *dev_res, *dev_a, *dev_b;
    cudaMalloc(&dev_res, N * sizeof(int));
    cudaMalloc(&dev_a, N * sizeof(int));
    cudaMalloc(&dev_b, N * sizeof(int));

    // Move input arrays onto GPU
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // Do computation.
    // <<<N,1>>> runs N copies of the kernel in parallel
    // But how does each copy of the kernel know which one it is?
    // Go look at the function!
    // <<<1,N>>> is, for this use case totally equivalent.
    // It just splits each exec into N threads in 1 block rather than
    // 1 thread in N blocks

    // add_block<<<N,1>>>(dev_res, dev_a, dev_b);
    add_thread<<<1,N>>>(dev_res, dev_a, dev_b);

    // Move output array back to CPU
    cudaMemcpy(res, dev_res, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Free everything on GPU
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_res);

    // Print results
    for (int i=0; i<N; i++) {
        printf("%d, ", res[i]);
    }
    printf("\n");

    return 0;
}
