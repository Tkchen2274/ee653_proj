#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

int main() {
    FILE *input_file = fopen("input.txt", "r");
    FILE *output_file = fopen("output.txt", "w");
    
    if (!input_file || !output_file) {
        fprintf(stderr, "Error: Could not open input.txt or output.txt\n");
        return 1;
    }

    mpz_t a, b, gcd, ba, bb;
    mpz_inits(a, b, gcd, ba, bb, NULL);
    struct timespec start, end;
    double elapsed_time;
    char a_str[1024], b_str[1024];
    char buffer[4096];
    int test_case = 0;
    double total_time = 0.0;

    // Process each line
    while (fscanf(input_file, "%1023s %1023s", a_str, b_str) == 2) {
        test_case++;
        
        // Convert strings to GMP integers
        mpz_set_str(a, a_str, 10);
        mpz_set_str(b, b_str, 10);

        // Start timing
        clock_gettime(CLOCK_MONOTONIC, &start);

        // Compute extended GCD
        xgcd_gmp(a, b, gcd, ba, bb);

        // End timing
        clock_gettime(CLOCK_MONOTONIC, &end);
        elapsed_time = (end.tv_sec - start.tv_sec) + 
                      (end.tv_nsec - start.tv_nsec) / 1e9;
        total_time += elapsed_time;

        // Print to console
        printf("\nTest Case %d:\n", test_case);
        printf("Execution time: %.9f seconds\n", elapsed_time);
        gmp_printf("a: %Zd\n", a);
        gmp_printf("b: %Zd\n", b);
        gmp_printf("GCD: %Zd\n", gcd);
        gmp_printf("Coefficient a: %Zd\n", ba);
        gmp_printf("Coefficient b: %Zd\n", bb);

        // Verify: ba*a + bb*b = gcd
        mpz_t verify;
        mpz_init(verify);
        mpz_mul(verify, ba, a);
        mpz_addmul(verify, bb, b);
        gmp_printf("Verification: %Zd\n", verify);
        
        // Write results to output file
        fprintf(output_file, "Case %d:\n", test_case);
        gmp_snprintf(buffer, 4096, "a: %Zd\n", a);
        fprintf(output_file, "%s", buffer);
        gmp_snprintf(buffer, 4096, "b: %Zd\n", b);
        fprintf(output_file, "%s", buffer);
        gmp_snprintf(buffer, 4096, "GCD: %Zd\n", gcd);
        fprintf(output_file, "%s", buffer);
        gmp_snprintf(buffer, 4096, "Coefficients (a,b): %Zd, %Zd\n", ba, bb);
        fprintf(output_file, "%s", buffer);
        fprintf(output_file, "Time: %.9f seconds\n\n", elapsed_time);

        mpz_clear(verify);
    }

    // Print average time
    double average_time = total_time / test_case;
    printf("\n=== Summary ===\n");
    printf("Total test cases: %d\n", test_case);
    printf("Total time: %.9f seconds\n", total_time);
    printf("Average time: %.9f seconds\n", average_time);

    // Write summary to output file
    fprintf(output_file, "=== Summary ===\n");
    fprintf(output_file, "Total test cases: %d\n", test_case);
    fprintf(output_file, "Total time: %.9f seconds\n", total_time);
    fprintf(output_file, "Average time: %.9f seconds\n", average_time);

    mpz_clears(a, b, gcd, ba, bb, NULL);
    fclose(input_file);
    fclose(output_file);
    return 0;
}
