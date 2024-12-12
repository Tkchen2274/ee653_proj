#include <gmp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * Extended GCD (Greatest Common Divisor) using GMP for 256-bit integers.
 *
 * @param a0 First input number
 * @param b0 Second input number
 * @param gcd Pointer to store the GCD result
 * @param ba Pointer to store Bezout coefficient for a0
 * @param bb Pointer to store Bezout coefficient for b0
 */
void xgcd_gmp(const mpz_t a0, const mpz_t b0, mpz_t gcd, mpz_t ba, mpz_t bb) {
    mpz_t u, v, x, y, a, b, q, r, temp;
    mpz_inits(u, v, x, y, a, b, q, r, temp, NULL);

    // Initialize variables
    mpz_set(a, a0);
    mpz_set(b, b0);
    mpz_set_ui(u, 1);
    mpz_set_ui(v, 0);
    mpz_set_ui(x, 0);
    mpz_set_ui(y, 1);

    // Extended Euclidean algorithm loop
    while (mpz_sgn(b) != 0) {
        mpz_fdiv_qr(q, r, a, b); // q = a / b, r = a % b

        // Update a and b
        mpz_set(a, b);
        mpz_set(b, r);

        // Update Bezout coefficients
        mpz_set(temp, x);
        mpz_mul(x, q, x);
        mpz_sub(x, u, x);
        mpz_set(u, temp);

        mpz_set(temp, y);
        mpz_mul(y, q, y);
        mpz_sub(y, v, y);
        mpz_set(v, temp);
    }

    // Set results
    mpz_set(gcd, a);
    mpz_set(ba, u);
    mpz_set(bb, v);

    // Clear memory
    mpz_clears(u, v, x, y, a, b, q, r, temp, NULL);
}

int main() {
    FILE *input_file = fopen("input.txt", "r");
    FILE *output_file = fopen("output.txt", "w");
    if (!input_file || !output_file) {
        fprintf(stderr, "Error: Could not open input.txt or output.txt\n");
        return 1;
    }

    mpz_t a, b, gcd, ba, bb;
    mpz_inits(a, b, gcd, ba, bb, NULL);

    char a_str[1024], b_str[1024];
    while (fscanf(input_file, "%1023s %1023s", a_str, b_str) == 2) {
        mpz_set_str(a, a_str, 10);
        mpz_set_str(b, b_str, 10);

        // Measure execution time
        clock_t start_time = clock();

        // Compute extended GCD
        xgcd_gmp(a, b, gcd, ba, bb);

        clock_t end_time = clock();
        double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

        // Output results to file
        char buffer[2048];
        gmp_sprintf(buffer, "%Zd %Zd %Zd %Zd %Zd %.6f\n", a, b, gcd, ba, bb, elapsed_time);
        fprintf(output_file, "%s\n", buffer);
    }

    fclose(input_file);
    fclose(output_file);

    // Clear memory
    mpz_clears(a, b, gcd, ba, bb, NULL);
    return 0;
}
