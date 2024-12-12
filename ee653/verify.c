#include <gmp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * Verifies if the GCD and Bézout coefficients satisfy Bézout's identity.
 *
 * @param a First integer
 * @param b Second integer
 * @param gcd Calculated GCD
 * @param ba Bézout coefficient for a
 * @param bb Bézout coefficient for b
 * @return True if values are valid, false otherwise
 */
bool verify_gcd_and_bezout(const mpz_t a, const mpz_t b, const mpz_t gcd, const mpz_t ba, const mpz_t bb) {
    mpz_t lhs, rhs;
    mpz_inits(lhs, rhs, NULL);

    // Calculate lhs = a * ba + b * bb
    mpz_mul(lhs, a, ba);
    mpz_addmul(lhs, b, bb);

    // Verify lhs == gcd
    bool is_valid = (mpz_cmp(lhs, gcd) == 0);

    mpz_clears(lhs, rhs, NULL);
    return is_valid;
}

int main() {
    FILE *output_file = fopen("output.txt", "r");
    if (!output_file) {
        fprintf(stderr, "Error: Could not open output.txt\n");
        return 1;
    }

    mpz_t a, b, gcd, ba, bb;
    mpz_inits(a, b, gcd, ba, bb, NULL);

    char a_str[1024], b_str[1024], gcd_str[1024], ba_str[1024], bb_str[1024];
    bool all_valid = true;

    while (fscanf(output_file, "%1023s %1023s %1023s %1023s %1023s", a_str, b_str, gcd_str, ba_str, bb_str) == 5) {
        // Consume the rest of the line to avoid the extra value being read
        char extra[1024];  // Buffer to hold the extra value (if any)
        if (fscanf(output_file, "%1023s", extra) == 1) {
            // Optionally print or handle the extra value here if needed
            // printf("Extra value: %s\n", extra);
        }
        // Set values from strings
        mpz_set_str(a, a_str, 10);
        mpz_set_str(b, b_str, 10);
        mpz_set_str(gcd, gcd_str, 10);
        mpz_set_str(ba, ba_str, 10);
        mpz_set_str(bb, bb_str, 10);

        // Verify GCD and Bézout coefficients
        if (!verify_gcd_and_bezout(a, b, gcd, ba, bb)) {
            printf("Invalid result for a=%s, b=%s\n", a_str, b_str);
            all_valid = false;
        }
        else{
            printf("Valid result for a=%s, b=%s, gcd=%s, ba=%s, bb=%s\n", a_str, b_str, gcd_str, ba_str, bb_str);
        }
    }

    fclose(output_file);
    mpz_clears(a, b, gcd, ba, bb, NULL);

    if (all_valid) {
        printf("All values in output.txt are correct.\n");
        return 0;
    } else {
        printf("Some values in output.txt are incorrect.\n");
        return 1;
    }
}
