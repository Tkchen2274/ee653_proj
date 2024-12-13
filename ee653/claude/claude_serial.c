#include <gmp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Structure to hold timing statistics
typedef struct {
    double total_time;
    double xgcd_time;
    unsigned long iterations;
} TimingStats;

/**
 * Extended GCD (Greatest Common Divisor) using GMP for 256-bit integers.
 * Returns Bezout coefficients and GCD.
 */
void xgcd_gmp(const mpz_t a0, const mpz_t b0, mpz_t gcd, mpz_t ba, mpz_t bb) {
    mpz_t u, v, x, y, a, b, q, r, temp;
    mpz_inits(u, v, x, y, a, b, q, r, temp, NULL);

    mpz_set(a, a0);
    mpz_set(b, b0);
    mpz_set_ui(u, 1);
    mpz_set_ui(v, 0);
    mpz_set_ui(x, 0);
    mpz_set_ui(y, 1);

    while (mpz_sgn(b) != 0) {
        mpz_fdiv_qr(q, r, a, b);
        mpz_set(a, b);
        mpz_set(b, r);

        mpz_set(temp, x);
        mpz_mul(x, q, x);
        mpz_sub(x, u, x);
        mpz_set(u, temp);

        mpz_set(temp, y);
        mpz_mul(y, q, y);
        mpz_sub(y, v, y);
        mpz_set(v, temp);
    }

    mpz_set(gcd, a);
    mpz_set(ba, u);
    mpz_set(bb, v);

    mpz_clears(u, v, x, y, a, b, q, r, temp, NULL);
}

/**
 * VDF computation function that implements the time-lock puzzle
 */
void compute_vdf(const mpz_t x, const mpz_t N, const mpz_t b0, unsigned long T,
                 mpz_t final_a, mpz_t final_r, mpz_t final_P, TimingStats* stats) {
    mpz_t a, gcd, ba, bb, r, y, P, temp;
    mpz_inits(a, gcd, ba, bb, r, y, P, temp, NULL);
    struct timespec xgcd_start, xgcd_end;

    // Initialize values
    mpz_set(a, x);
    mpz_set_ui(r, 1);
    mpz_set_ui(y, 1);
    mpz_set_ui(P, 1);

    stats->iterations = T;
    stats->xgcd_time = 0.0;

    // Main VDF loop
    for (unsigned long i = 0; i < T; i++) {
        // Time the XGCD operation
        clock_gettime(CLOCK_MONOTONIC, &xgcd_start);
        xgcd_gmp(a, b0, gcd, ba, bb);
        clock_gettime(CLOCK_MONOTONIC, &xgcd_end);
        
        // Accumulate XGCD time
        stats->xgcd_time += (xgcd_end.tv_sec - xgcd_start.tv_sec) +
                           (xgcd_end.tv_nsec - xgcd_start.tv_nsec) / 1e9;

        // Check if ba is even (simple selection criteria)
        if (mpz_even_p(ba)) {
            // Update y and P
            mpz_set(y, ba);
            mpz_mul(P, P, y);
            mpz_mod(P, P, N);
            
            // Update r
            mpz_mul(r, r, y);
            mpz_mod(r, r, N);
        }

        // Update a = (ba*x + bb*b0) mod N
        mpz_mul(temp, ba, x);
        mpz_addmul(temp, bb, b0);
        mpz_mod(a, temp, N);
    }

    // Set final values
    mpz_set(final_a, a);
    mpz_set(final_r, r);
    mpz_set(final_P, P);

    mpz_clears(a, gcd, ba, bb, r, y, P, temp, NULL);
}

int main() {
    FILE *input_file = fopen("input.txt", "r");
    FILE *output_file = fopen("output.txt", "w");
    FILE *timing_file = fopen("timing_analysis.txt", "w");
    
    if (!input_file || !output_file || !timing_file) {
        fprintf(stderr, "Error: Could not open input.txt, output.txt, or timing_analysis.txt\n");
        return 1;
    }

    // Write headers for timing analysis
    fprintf(timing_file, "Total_Time(s),XGCD_Time(s),XGCD_Percentage,Iterations\n");

    mpz_t x, N, b0, final_a, final_r, final_P;
    mpz_inits(x, N, b0, final_a, final_r, final_P, NULL);
    unsigned long T;
    TimingStats stats = {0.0, 0.0, 0};

    // Read input format: x N b0 T
    char x_str[1024], N_str[1024], b0_str[1024];
    int test_case = 0;
    while (fscanf(input_file, "%1023s %1023s %1023s %lu", x_str, N_str, b0_str, &T) == 4) {
        test_case++;
        mpz_set_str(x, x_str, 10);
        mpz_set_str(N, N_str, 10);
        mpz_set_str(b0, b0_str, 10);

        // Measure total execution time
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        // Compute VDF
        compute_vdf(x, N, b0, T, final_a, final_r, final_P, &stats);

        clock_gettime(CLOCK_MONOTONIC, &end);
        stats.total_time = (end.tv_sec - start.tv_sec) + 
                          (end.tv_nsec - start.tv_nsec) / 1e9;

        // Write timing analysis
        double xgcd_percentage = (stats.xgcd_time / stats.total_time) * 100.0;
        fprintf(timing_file, "%.6f,%.6f,%.2f%%,%lu\n", 
                stats.total_time, stats.xgcd_time, xgcd_percentage, stats.iterations);

        // Print analysis to console
        printf("\nTest Case %d Analysis:\n", test_case);
        printf("Total Time: %.6f seconds\n", stats.total_time);
        printf("XGCD Time: %.6f seconds\n", stats.xgcd_time);
        printf("XGCD Percentage: %.2f%%\n", xgcd_percentage);
        printf("Iterations: %lu\n\n", stats.iterations);
        char buffer[2048];
        // Output results: final_a final_r final_P elapsed_time
        gmp_sprintf(buffer, "%Zd %Zd %Zd %.6f\n", 
                   final_a, final_r, final_P, stats.total_time);
    }

    fclose(input_file);
    fclose(output_file);
    fclose(timing_file);
    mpz_clears(x, N, b0, final_a, final_r, final_P, NULL);
    return 0;
}
