#include <stdio.h>

int main(void) {
    // Return info about the 0th device
    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, 0);


    printf("Name:                   %s\n", deviceProperties.name);
    printf("Total mem:              %luMB\n", deviceProperties.totalGlobalMem/1024/1024);
    printf("Max threads per block:  %d\n", deviceProperties.maxThreadsPerBlock);
    printf("Single to double perf:  %d\n", deviceProperties.singleToDoublePrecisionPerfRatio);
    return 0;
}
