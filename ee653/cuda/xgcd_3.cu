#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <stdint.h>

#define MAX_DIGITS 8  // 256 bits = 8 * 32-bit digits

void print_number(const char *label, uint32_t *num) {
    printf("%s: 0x", label);
    for (int i = MAX_DIGITS - 1; i >= 0; i--) {
        printf("%08x", num[i]);
    }
    printf("\n");
}

__device__ bool is_zero(uint32_t *a) {
    for (int i = 0; i < MAX_DIGITS; i++) {
        if (a[i] != 0) return false;
    }
    return true;
}

__device__ void copy(uint32_t *dst, uint32_t *src) {
    for (int i = 0; i < MAX_DIGITS; i++) {
        dst[i] = src[i];
    }
}

__device__ bool greater_than_or_equal(uint32_t *a, uint32_t *b) {
    for (int i = MAX_DIGITS - 1; i >= 0; i--) {
        if (a[i] > b[i]) return true;
        if (a[i] < b[i]) return false;
    }
    return true;
}

__device__ void subtract_device(uint32_t *a, uint32_t *b, uint32_t *result) {
    uint64_t borrow = 0;
    for (int i = 0; i < MAX_DIGITS; i++) {
        uint64_t diff = (uint64_t)a[i] - (uint64_t)b[i] - borrow;
        result[i] = (uint32_t)diff;
        borrow = ((diff >> 32) & 1);
    }
}

__global__ void gcd_kernel(uint32_t *a, uint32_t *b, uint32_t *result, uint32_t *debug_flag) {
    // Local copies of inputs
    uint32_t u[MAX_DIGITS], v[MAX_DIGITS];
    copy(u, a);
    copy(v, b);
    
    // Main GCD loop
    while (!is_zero(v)) {
        if (greater_than_or_equal(u, v)) {
            uint32_t temp[MAX_DIGITS];
            subtract_device(u, v, temp);
            copy(u, temp);
        } else {
            uint32_t temp[MAX_DIGITS];
            subtract_device(v, u, temp);
            copy(v, temp);
        }
        
        atomicAdd(debug_flag, 1);
    }
    
    copy(result, u);
}

int main() {
    // Example: Let's use small numbers first: a = 48 and b = 18
    uint32_t h_a[MAX_DIGITS] = {48, 0, 0, 0, 0, 0, 0, 0};  // Just the first digit is 48
    uint32_t h_b[MAX_DIGITS] = {18, 0, 0, 0, 0, 0, 0, 0};  // Just the first digit is 18
    uint32_t h_result[MAX_DIGITS] = {0};
    uint32_t h_debug_flag = 0;

    printf("Test values:\n");
    print_number("a", h_a);
    print_number("b", h_b);

    uint32_t *d_a, *d_b, *d_result, *d_debug_flag;
    cudaMalloc(&d_a, MAX_DIGITS * sizeof(uint32_t));
    cudaMalloc(&d_b, MAX_DIGITS * sizeof(uint32_t));
    cudaMalloc(&d_result, MAX_DIGITS * sizeof(uint32_t));
    cudaMalloc(&d_debug_flag, sizeof(uint32_t));

    cudaMemcpy(d_a, h_a, MAX_DIGITS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, MAX_DIGITS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_debug_flag, &h_debug_flag, sizeof(uint32_t), cudaMemcpyHostToDevice);

    printf("Launching kernel...\n");
    gcd_kernel<<<1, 1>>>(d_a, d_b, d_result, d_debug_flag);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_result, d_result, MAX_DIGITS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_debug_flag, d_debug_flag, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("Kernel completed with %u iterations\n", h_debug_flag);
    print_number("Final GCD", h_result);

    // Expected result for GCD(48, 18) should be 6
    printf("Note: For input values 48 and 18, expected GCD is 6\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    cudaFree(d_debug_flag);

    return 0;
}
