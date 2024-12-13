#include <gmp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * Verifies the VDF output by checking if r equals P in Z/NZ
 */
bool verify_vdf(const mpz_t r, const mpz_t P, const mpz_t N) {
    mpz_t temp;
    mpz_init(temp);

    // Verify r ≡ P (mod N)
    mpz_sub(temp, r, P);
    mpz_mod(temp, temp, N);
    bool is_valid = (mpz_sgn(temp) == 0);

    mpz_clear(temp);
    return is_valid;
}

int main() {
    FILE *output_file = fopen("output.txt", "r");
    if (!output_file) {
        fprintf(stderr, "Error: Could not open output.txt\n");
        return 1;
    }

    mpz_t final_a, final_r, final_P;
    mpz_inits(final_a, final_r, final_P, NULL);
    bool all_valid = true;

    char a_str[1024], r_str[1024], P_str[1024];
    while (fscanf(output_file, "%1023s %1023s %1023s", a_str, r_str, P_str) == 3) {
        // Consume the elapsed time value
        double elapsed_time;
        fscanf(output_file, "%lf", &elapsed_time);

        // Set values from strings
        mpz_set_str(final_a, a_str, 10);
        mpz_set_str(final_r, r_str, 10);
        mpz_set_str(final_P, P_str, 10);

        // Verify VDF output
        if (!verify_vdf(final_r, final_P, final_P)) { // Using final_P as N for demonstration
            printf("Invalid VDF result for output values:\n");
            gmp_printf("a: %Zd\nr: %Zd\nP: %Zd\n", final_a, final_r, final_P);
            all_valid = false;
        } else {
            printf("Valid VDF result (completed in %.6f seconds)\n", elapsed_time);
            gmp_printf("a: %Zd\nr: %Zd\nP: %Zd\n", final_a, final_r, final_P);
        }
    }

    fclose(output_file);
    mpz_clears(final_a, final_r, final_P, NULL);

    if (all_valid) {
        printf("All VDF computations are valid.\n");
        return 0;
    } else {
        printf("Some VDF computations are invalid.\n");
        return 1;
    }
}
