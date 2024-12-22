#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <gmp.h>
#include <pthread.h>

#define PORT 8080
#define MAX_CLIENTS 10
#define ITERATIONS 1000000 // Number of VDF iterations

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
void compute_vdf(const mpz_t x, const mpz_t a0, const mpz_t b0, unsigned long iterations,
                 mpz_t final_a, mpz_t final_r, mpz_t final_P, mpz_t prev_r, mpz_t prev_P, TimingStats* stats) {
    mpz_t gcd, ba, bb, r, y, P, temp, a;
    mpz_inits(gcd, ba, bb, r, y, P, temp, a, NULL);
    struct timespec xgcd_start, xgcd_end;

    // Initialize values
    mpz_set(a, x);
    mpz_set_ui(r, 1);
    mpz_set_ui(y, 1);
    mpz_set_ui(P, 1);

    // Initialize prev_r and prev_P
    mpz_set_ui(prev_r, 0);
    mpz_set_ui(prev_P, 0);

    stats->iterations = iterations;
    stats->xgcd_time = 0.0;

    // Main VDF loop
    for (unsigned long i = 0; i < iterations; i++) {
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
            mpz_mod(P, P, x); // Use x as the modulus

            // Update r
            mpz_mul(r, r, y);
            mpz_mod(r, r, x); // Use x as the modulus
        }

        // Update a = (ba*x + bb*b0) mod x
        mpz_mul(temp, ba, x);
        mpz_addmul(temp, bb, b0);
        mpz_mod(a, temp, x); // Use x as the modulus

        // Store the second-to-last values of r and P
        if (i < iterations - 1) {
            mpz_set(prev_r, r);
            mpz_set(prev_P, P);
        }
    }

    // Set final values
    mpz_set(final_a, a);
    mpz_set(final_r, r);
    mpz_set(final_P, P);

    mpz_clears(gcd, ba, bb, r, y, P, temp, a, NULL);
}

int main(int argc, char const *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <server_ip>\n", argv[0]);
        return 1;
    }

    int sock = 0, valread;
    struct sockaddr_in serv_addr;
    char buffer[4096] = {0};

    // Create socket
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("\n Socket creation error \n");
        return -1;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);

    // Convert IPv4 and IPv6 addresses from text to binary form
    if (inet_pton(AF_INET, argv[1], &serv_addr.sin_addr) <= 0) {
        printf("\nInvalid address/ Address not supported \n");
        return -1;
    }

    // Connect to server
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        printf("\nConnection Failed \n");
        return -1;
    }

    // Send request to server
    send(sock, "REQUEST", strlen("REQUEST"), 0);

    // Receive VDF puzzle
    valread = read(sock, buffer, 4096);
    if (valread <= 0) {
        perror("read failed");
        close(sock);
        return -1;
    }

    printf("Received from server: %s\n", buffer);

    if (strncmp(buffer, "PUZZLE", 6) == 0) {
        // Parse puzzle parameters: "PUZZLE x a b T"
        mpz_t x, a, b, final_a, final_r, final_P, prev_r, prev_P;
        mpz_inits(x, a, b, final_a, final_r, final_P, prev_r, prev_P, NULL);
        unsigned long T;

        char x_str[1024], a_str[1024], b_str[1024];
        if (sscanf(buffer, "PUZZLE %s %s %s %lu", x_str, a_str, b_str, &T) != 4) {
            fprintf(stderr, "Error parsing PUZZLE message\n");
            mpz_clears(x, a, b, final_a, final_r, final_P, prev_r, prev_P, NULL);
            close(sock);
            return 1;
        }

        // Set the values of x, a, and b from the parsed strings
        mpz_set_str(x, x_str, 10);
        mpz_set_str(a, a_str, 10);
        mpz_set_str(b, b_str, 10);

        // Compute VDF
        TimingStats stats = {0.0, 0.0, 0};
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        compute_vdf(x, a, b, T, final_a, final_r, final_P, prev_r, prev_P, &stats);
        clock_gettime(CLOCK_MONOTONIC, &end);
        stats.total_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        // Send solution to server: "SOLUTION prev_r prev_P"
        char solution_msg[4096];
        gmp_sprintf(solution_msg, "SOLUTION %Zd %Zd", prev_r, prev_P);
        send(sock, solution_msg, strlen(solution_msg), 0);

        // Receive and print data or error message
        memset(buffer, 0, sizeof(buffer));
        valread = read(sock, buffer, 4096);
        printf("Server response: %.*s\n", valread, buffer);

        mpz_clears(x, a, b, final_a, final_r, final_P, prev_r, prev_P, NULL);
    } else {
        // Server sent an error message
        printf("Server: %s\n", buffer);
    }

    close(sock);
    return 0;
}
