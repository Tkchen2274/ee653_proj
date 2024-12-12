#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h> 

/**
 * Extended GCD (Greatest Common Divisor) function
 * 
 * @param a0 First input number
 * @param b0 Second input number
 * @param constant_time Whether to use constant-time algorithm
 * @param bitwidth Maximum bitwidth of inputs
 * @param gcd Pointer to store the GCD result
 * @param ba Pointer to store Bezout coefficient for a0
 * @param bb Pointer to store Bezout coefficient for b0
 */

// Helper function for xgcd update similar to Python implementation
void xgcd_update(int num_bits_reduced, int64_t* u_ptr, int64_t* m_ptr, int64_t bm, int64_t am) {
    for (int i = 0; i < num_bits_reduced; i++) {
        if (*u_ptr % 2 == 1) {
            *u_ptr = (*u_ptr + bm) / 2;
            *m_ptr = (*m_ptr - am) / 2;
        } else {
            *u_ptr = *u_ptr / 2;
            *m_ptr = *m_ptr / 2;
        }
    }
}

void xgcd(int64_t a0, int64_t b0, bool constant_time, int bitwidth, 
          int64_t* gcd, int64_t* ba, int64_t* bb) {
    // Step 1: Pre-processing
    int64_t am, bm;
    if (a0 % 2 == 0) {
        am = a0 + b0;
        bm = b0;
    } else if (b0 % 2 == 0) {
        am = a0;
        bm = a0 + b0;
    } else {
        am = a0;
        bm = b0;
    }

    int64_t a = am, b = bm;
    int64_t u = 1, m = 0, y = 0, n = 1;
    int64_t delta = 0;
    int iterations = 0;
    bool end_loop = false;

    
    // Step 2: Iteration loop
    while (!end_loop) {
        if (!constant_time && (a % 8 == 0)) {
            a = a / 8;
            delta -= 3;
            xgcd_update(3, &u, &m, bm, am);
        } else if (!constant_time && (a % 4 == 0)) {
            a = a / 4;
            delta -= 2;
            xgcd_update(2, &u, &m, bm, am);
        } else if (a % 2 == 0) {
            a = a / 2;
            delta -= 1;
            xgcd_update(1, &u, &m, bm, am);
        } else if (!constant_time && (b % 8 == 0)) {
            b = b / 8;
            delta += 3;
            xgcd_update(3, &y, &n, bm, am);
        } else if (!constant_time && (b % 4 == 0)) {
            b = b / 4;
            delta += 2;
            xgcd_update(2, &y, &n, bm, am);
        } else if (b % 2 == 0) {
            b = b / 2;
            delta += 1;
            xgcd_update(1, &y, &n, bm, am);
        } else if ((delta >= 0) && ((b + a) % 4 == 0)) {
            a = (a + b) / 4;
            delta -= 1;
            int64_t new_u = u + y;
            int64_t new_m = m + n;
            xgcd_update(2, &new_u, &new_m, bm, am);
            u = new_u;
            m = new_m;
        } else if ((delta >= 0) && ((b - a) % 4 == 0)) {
            a = (a - b) / 4;
            delta -= 1;
            int64_t new_u = u - y;
            int64_t new_m = m - n;
            xgcd_update(2, &new_u, &new_m, bm, am);
            u = new_u;
            m = new_m;
        } else if ((delta < 0) && ((b + a) % 4 == 0)) {
            b = (a + b) / 4;
            delta += 1;
            int64_t new_y = u + y;
            int64_t new_n = m + n;
            xgcd_update(2, &new_y, &new_n, bm, am);
            y = new_y;
            n = new_n;
        } else {
            b = (a - b) / 4;
            delta += 1;
            int64_t new_y = u - y;
            int64_t new_n = m - n;
            xgcd_update(2, &new_y, &new_n, bm, am);
            y = new_y;
            n = new_n;
        }

        // Termination condition
        if (constant_time) {
            iterations++;
            end_loop = (iterations >= 1.51 * bitwidth + 1);
        } else {
            end_loop = (a == 0 || b == 0);
        }
    }

    // Step 3: Post-processing
    int64_t result_gcd = a + b;
    u = u + y;
    m = m + n;

    if (a0 % 2 == 0) {
        m = u + m;
    } else if (b0 % 2 == 0) {
        u = u + m;
    }

    if (result_gcd < 0) {
        result_gcd = -result_gcd;
        u = -u;
        m = -m;
    }

    // Set output parameters
    *gcd = result_gcd;
    *ba = u;
    *bb = m;
}

// Example usage


int main() {
    // Open the file for reading
    FILE *file = fopen("input.txt", "r");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    // Open the output file for writing
    FILE *output = fopen("output.txt", "w");
    if (output == NULL) {
        perror("Error opening output file");
        fclose(file);
        return 1;
    }

    int64_t a0, b0;
    int64_t gcd, ba, bb;
    struct timespec start, end;
    double elapsed_time;

    printf("Processing inputs from file:\n\n");

    // Read and process all pairs of numbers until EOF
    while (fscanf(file, "%ld %ld", &a0, &b0) == 2) {

        // Get the start time
        if (clock_gettime(CLOCK_REALTIME, &start) == -1) {
            perror("Error getting start time");
            fclose(file);
            return 1;
        }
        xgcd(a0, b0, 0, 256, &gcd, &ba, &bb);

        // Get the end time
        if (clock_gettime(CLOCK_REALTIME, &end) == -1) {
            perror("Error getting end time");
            fclose(file);
            return 1;
        }

        elapsed_time = (end.tv_sec - start.tv_sec) +
                              (end.tv_nsec - start.tv_nsec) / 1e9;

        // Write results to both standard output and the output file
        // fprintf(output, "GCD(%ld, %ld) = %ld\n", a0, b0, gcd);
        // fprintf(output, "Bezout Coefficients: %ld * %ld + %ld * %ld = %ld\n",
        //        ba, a0, bb, b0, gcd);
        fprintf(output,"%.9f\n", elapsed_time);

        printf("GCD(%ld, %ld) = %ld\n", a0, b0, gcd);
        printf("Bezout Coefficients: %ld * %ld + %ld * %ld = %ld\n\n", 
               ba, a0, bb, b0, gcd);
        printf("Time taken: %.9f seconds\n\n", elapsed_time);
        // printf("__________________________\n");
    }

    fclose(file);
    fclose(output);
    return 0;
}

// int main() {
//     // int64_t a0 = 48, b0 = 18;
//     int64_t a0 = 81, b0 = 99;
//     int64_t gcd, ba, bb;
    
//     xgcd(a0, b0, false, 64, &gcd, &ba, &bb);
    
//     printf("GCD(%ld, %ld) = %ld\n", a0, b0, gcd);
//     printf("Bezout Coefficients: %ld * %ld + %ld * %ld = %ld\n", 
//            ba, a0, bb, b0, gcd);
    
//     return 0;
// }
