#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

__global__ void xgcd_kernel(const unsigned int* a, const unsigned int* b, 
                           unsigned int* gcd, int* ba, int* bb, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned int a_local = a[idx];
    unsigned int b_local = b[idx];
    int u = 1, v = 0, x = 0, y = 1;
    unsigned int temp;

    while (b_local != 0) {
        unsigned int q = a_local / b_local;
        unsigned int r = a_local % b_local;
        a_local = b_local;
        b_local = r;

        temp = x;
        x = u - (int)q * x;
        u = temp;

        temp = y;
        y = v - (int)q * y;
        v = temp;
    }

    gcd[idx] = a_local;
    ba[idx] = u;
    bb[idx] = v;
}

int read_input_file(const char* filename, unsigned int** a, unsigned int** b) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Could not open input file\n");
        return -1;
    }

    int count = 0;
    unsigned int temp1, temp2;
    while (fscanf(file, "%u %u", &temp1, &temp2) == 2) {
        count++;
    }

    *a = (unsigned int*)malloc(count * sizeof(unsigned int));
    *b = (unsigned int*)malloc(count * sizeof(unsigned int));
    if (!*a || !*b) {
        printf("Error: Memory allocation failed\n");
        fclose(file);
        return -1;
    }

    rewind(file);
    for (int i = 0; i < count; i++) {
        fscanf(file, "%u %u", &(*a)[i], &(*b)[i]);
    }

    fclose(file);
    return count;
}

int main() {
    const int NUM_RUNS = 100; // Number of times to run the test
    unsigned int *h_a = NULL, *h_b = NULL;
    int n = read_input_file("input.txt", &h_a, &h_b);
    if (n <= 0) return 1;

    // Allocate device memory
    unsigned int *d_a, *d_b, *d_gcd;
    int *d_ba, *d_bb;
    cudaMalloc(&d_a, n * sizeof(unsigned int));
    cudaMalloc(&d_b, n * sizeof(unsigned int));
    cudaMalloc(&d_gcd, n * sizeof(unsigned int));
    cudaMalloc(&d_ba, n * sizeof(int));
    cudaMalloc(&d_bb, n * sizeof(int));

    // Allocate host memory for results
    unsigned int *h_gcd = (unsigned int*)malloc(n * sizeof(unsigned int));
    int *h_ba = (int*)malloc(n * sizeof(int));
    int *h_bb = (int*)malloc(n * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_a, h_a, n * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Set up timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Array to store times
    float times[NUM_RUNS];
    float total_time = 0.0f;

    // Run multiple times
    for (int run = 0; run < NUM_RUNS; run++) {
        cudaEventRecord(start);
        xgcd_kernel<<<blocks, threadsPerBlock>>>(d_a, d_b, d_gcd, d_ba, d_bb, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        times[run] = milliseconds;
        total_time += milliseconds;
    }

    // Calculate statistics
    float avg_time = total_time / NUM_RUNS;
    
    // Calculate standard deviation
    float sum_squared_diff = 0.0f;
    for (int i = 0; i < NUM_RUNS; i++) {
        float diff = times[i] - avg_time;
        sum_squared_diff += diff * diff;
    }
    float std_dev = sqrtf(sum_squared_diff / NUM_RUNS);

    // Find min and max times
    float min_time = times[0], max_time = times[0];
    for (int i = 1; i < NUM_RUNS; i++) {
        if (times[i] < min_time) min_time = times[i];
        if (times[i] > max_time) max_time = times[i];
    }

    // Copy back results from last run for verification
    cudaMemcpy(h_gcd, d_gcd, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ba, d_ba, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bb, d_bb, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Write timing results
    FILE* output = fopen("timing_results.txt", "w");
    fprintf(output, "Performance Analysis (%d runs):\n", NUM_RUNS);
    fprintf(output, "Number of pairs processed per run: %d\n", n);
    fprintf(output, "Average time: %.6f ms\n", avg_time);
    fprintf(output, "Standard deviation: %.6f ms\n", std_dev);
    fprintf(output, "Minimum time: %.6f ms\n", min_time);
    fprintf(output, "Maximum time: %.6f ms\n", max_time);
    fprintf(output, "Average time per pair: %.6f ms\n", avg_time/n);
    
    // Also print to console
    printf("Performance Analysis (%d runs):\n", NUM_RUNS);
    printf("Number of pairs processed per run: %d\n", n);
    printf("Average time: %.6f ms\n", avg_time);
    printf("Standard deviation: %.6f ms\n", std_dev);
    printf("Minimum time: %.6f ms\n", min_time);
    printf("Maximum time: %.6f ms\n", max_time);
    // printf("Average time per pair: %.6f ms\n", avg_time/n);

    // Clean up
    fclose(output);
    free(h_a);
    free(h_b);
    free(h_gcd);
    free(h_ba);
    free(h_bb);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_gcd);
    cudaFree(d_ba);
    cudaFree(d_bb);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
