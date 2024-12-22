#include <stdio.h>
#include <cuda.h>

#define MAX_DIGITS 8  // 256 bits = 8 * 32-bit digits


// Helper function implementations
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

__device__ void subtract(uint32_t *a, uint32_t *b, uint32_t *res) {
    int64_t borrow = 0;
    for (int i = 0; i < MAX_DIGITS; i++) {
        int64_t diff = (int64_t)a[i] - b[i] - borrow;
        if (diff < 0) {
            diff += ((uint64_t)1 << 32);
            borrow = 1;
        } else {
            borrow = 0;
        }
        res[i] = (uint32_t)diff;
    }
}

__device__ void add(uint32_t *a, uint32_t *b, uint32_t *res) {
    uint64_t carry = 0;
    for (int i = 0; i < MAX_DIGITS; i++) {
        uint64_t sum = (uint64_t)a[i] + b[i] + carry;
        res[i] = (uint32_t)sum;
        carry = sum >> 32;
    }
}

__device__ void multiply(uint32_t *a, uint32_t *b, uint32_t *res) {
    uint64_t carry = 0;
    for (int i = 0; i < MAX_DIGITS; i++) {
        uint64_t product = (uint64_t)a[i] * b[i] + carry;
        res[i] = (uint32_t)product;
        carry = product >> 32;
    }
}


__device__ void divide(uint32_t *a, uint32_t *b, uint32_t *q, uint32_t *r) {
    copy(r, a);
    for (int i = 0; i < MAX_DIGITS; i++) q[i] = 0;

    while (!is_zero(r)) {
        if (r[MAX_DIGITS - 1] >= b[MAX_DIGITS - 1]) {
            q[0] += 1;
            subtract(r, b, r);
        } else {
            break;
        }
    }
}



// Kernel for Extended GCD
__global__ void xgcd_kernel(uint32_t *a, uint32_t *b, uint32_t *x, uint32_t *y, uint32_t *gcd) {
    uint32_t r1[MAX_DIGITS], r2[MAX_DIGITS];
    uint32_t s1[MAX_DIGITS], s2[MAX_DIGITS];
    uint32_t t1[MAX_DIGITS], t2[MAX_DIGITS];

    // Copy a and b into r1 and r2
    for (int i = 0; i < MAX_DIGITS; i++) {
        r1[i] = a[i];
        r2[i] = b[i];
        s1[i] = (i == 0 ? 1 : 0);
        s2[i] = 0;
        t1[i] = 0;
        t2[i] = (i == 0 ? 1 : 0);
    }

    uint32_t tmp[MAX_DIGITS];

    while (!is_zero(r2)) {
        uint32_t q[MAX_DIGITS];
        divide(r1, r2, q, tmp); // q = r1 / r2, tmp = r1 % r2

        copy(r1, r2);
        copy(r2, tmp);

        multiply(q, s2, tmp);
        subtract(s1, tmp, tmp);
        copy(s1, s2);
        copy(s2, tmp);

        multiply(q, t2, tmp);
        subtract(t1, tmp, tmp);
        copy(t1, t2);
        copy(t2, tmp);
    }

    for (int i = 0; i < MAX_DIGITS; i++) {
        gcd[i] = r1[i];
        x[i] = s1[i];
        y[i] = t1[i];
    }
}


int main() {
    uint32_t a[MAX_DIGITS] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                              0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
    uint32_t b[MAX_DIGITS] = {0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};

    uint32_t *d_a, *d_b, *d_x, *d_y, *d_gcd;
    uint32_t x[MAX_DIGITS], y[MAX_DIGITS], gcd[MAX_DIGITS];

    cudaMalloc(&d_a, MAX_DIGITS * sizeof(uint32_t));
    cudaMalloc(&d_b, MAX_DIGITS * sizeof(uint32_t));
    cudaMalloc(&d_x, MAX_DIGITS * sizeof(uint32_t));
    cudaMalloc(&d_y, MAX_DIGITS * sizeof(uint32_t));
    cudaMalloc(&d_gcd, MAX_DIGITS * sizeof(uint32_t));

    cudaMemcpy(d_a, a, MAX_DIGITS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, MAX_DIGITS * sizeof(uint32_t), cudaMemcpyHostToDevice);

    xgcd_kernel<<<1, 1>>>(d_a, d_b, d_x, d_y, d_gcd);

    cudaMemcpy(x, d_x, MAX_DIGITS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(y, d_y, MAX_DIGITS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(gcd, d_gcd, MAX_DIGITS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("GCD: ");
    for (int i = MAX_DIGITS - 1; i >= 0; i--) {
        printf("%08x", gcd[i]);
    }
    printf("\n");

    printf("x: ");
    for (int i = MAX_DIGITS - 1; i >= 0; i--) {
        printf("%08x", x[i]);
    }
    printf("\n");

    printf("y: ");
    for (int i = MAX_DIGITS - 1; i >= 0; i--) {
        printf("%08x", y[i]);
    }
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_gcd);

    return 0;
}
