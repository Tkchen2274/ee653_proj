#include <stdio.h>
__global__ void hello_cuda() {
	printf("Hello from CUDA! Thread %d in block %d\n", threadIdx.x, blockIdx.x);
}
int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    hello_cuda<<<1, 10>>>();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed time: %.3f ms\n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
