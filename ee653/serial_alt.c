#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

/**
 
Extended GCD (Greatest Common Divisor) function*
Computes gcd(a0, b0) along with Bezout coefficients x and y such that:
gcd(a0, b0) = x * a0 + y * b0*
@param a0 First input number
@param b0 Second input number
@param gcd Pointer to store the GCD result
@param x Pointer to store Bezout coefficient for a0
@param y Pointer to store Bezout coefficient for b0*/
void xgcd_alternate(int64_t a0, int64_t b0, int64_t* gcd, int64_t* x, int64_t* y) {
    int64_t a = a0, b = b0;
    int64_t x0 = 1, y0 = 0; // Coefficients for a
    int64_t x1 = 0, y1 = 1; // Coefficients for b

    while (b != 0) {
        int64_t q = a / b; // Quotient
        int64_t r = a % b; // Remainder

        // Update (a, b)
        a = b;
        b = r;

        // Update Bezout coefficients
        int64_t temp_x = x0 - q * x1;
        int64_t temp_y = y0 - q * y1;

        x0 = x1;
        y0 = y1;

        x1 = temp_x;
        y1 = temp_y;
    }

    // Set results
    *gcd = a;
    *x = x0;
    *y = y0;
}

// Example usage
int main() {
    int64_t a0, b0;
    int64_t gcd, x, y;

    printf("Enter two integers: ");
    //if (scanf("%ld %ld", &a0, &b0) != 2) {
    //     fprintf(stderr, "Invalid input!\n");
    //     return 1;
    // }
    a0 = 2001;
    b0 = 4071;

    xgcd_alternate(a0, b0, &gcd, &x, &y);

    printf("GCD(%ld, %ld) = %ld\n", a0, b0, gcd);
    printf("Bezout Coefficients: %ld %ld + %ld * %ld = %ld\n",
           x, a0, y, b0, gcd);

    return 0;
}
