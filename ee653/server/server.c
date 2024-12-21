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
#include <errno.h>

#define PORT 8080
#define MAX_CLIENTS 10
#define NUM_ITERATIONS 1000000

// Client status structure
typedef struct {
    char ip_address[INET_ADDRSTRLEN];
    bool solving;
    time_t challenge_start_time;
    mpz_t x, a, b, expected_r, expected_P; // VDF parameters
} ClientStatus;

ClientStatus clients[MAX_CLIENTS];
pthread_mutex_t clients_mutex = PTHREAD_MUTEX_INITIALIZER;

// Function to read data from data.txt (for authorized clients)
char* read_data() {
    printf("Entering read_data()\n");
    FILE* fp = fopen("data.txt", "r");
    if (fp == NULL) {
        perror("Error opening data.txt");
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char* data = malloc(fsize + 1);
    if (data == NULL) {
        perror("Memory allocation failed in read_data()");
        fclose(fp);
        return NULL;
    }

    fread(data, 1, fsize, fp);
    fclose(fp);

    data[fsize] = 0;
    printf("Exiting read_data()\n");
    return data;
}

// Function to generate random 256-bit numbers for VDF parameters
void generate_vdf_parameters(mpz_t x, mpz_t a, mpz_t b) {
    printf("Entering generate_vdf_parameters()\n");
    gmp_randstate_t state;
    gmp_randinit_default(state);
    gmp_randseed_ui(state, time(NULL)); // Seed with current time

    // Initialize x, a, and b
    mpz_init(x);
    mpz_init(a);
    mpz_init(b);

    // Generate random numbers (no need for primes)
    mpz_urandomb(x, state, 256); // 256-bit random number for x
    mpz_urandomb(a, state, 256); // 256-bit random number for a
    mpz_urandomb(b, state, 256); // 256-bit random number for b

    gmp_printf("Generated x: %Zd\n", x);
    gmp_printf("Generated a: %Zd\n", a);
    gmp_printf("Generated b: %Zd\n", b);

    gmp_randclear(state);
    printf("Exiting generate_vdf_parameters()\n");
}

// Function to verify the VDF solution
bool verify_vdf(const mpz_t r, const mpz_t P, const mpz_t a, const mpz_t b) {
    printf("Entering verify_vdf()\n");
    mpz_t temp;
    mpz_init(temp);

    // Replace this with your actual VDF verification logic
    // This is just a placeholder for demonstration
    // In a real VDF, the verification would involve checking if 'r' and 'P'
    // are the correct result of the VDF computation based on 'a' and 'b'.
    bool is_valid = true; // Replace with actual verification

    mpz_clear(temp);
    printf("Exiting verify_vdf()\n");
    return is_valid;
}

// Thread function for handling client requests
void* handle_client(void* arg) {
    printf("Entering handle_client()\n");
    int client_socket = *(int*)arg;
    char client_ip[INET_ADDRSTRLEN];
    struct sockaddr_in client_addr;
    socklen_t addr_len = sizeof(client_addr);

    // Get client IP address
    getpeername(client_socket, (struct sockaddr*)&client_addr, &addr_len);
    inet_ntop(AF_INET, &(client_addr.sin_addr), client_ip, INET_ADDRSTRLEN);

    printf("Client connected: %s\n", client_ip);

    // Find client in the list or add if new
    pthread_mutex_lock(&clients_mutex);
    printf("handle_client: Acquired clients_mutex\n");
    int client_index = -1;
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (strcmp(clients[i].ip_address, client_ip) == 0) {
            client_index = i;
            printf("Client %s found at index %d\n", client_ip, client_index);
            break;
        } else if (strlen(clients[i].ip_address) == 0) {
            client_index = i;
            strcpy(clients[i].ip_address, client_ip);
            printf("Client %s added at index %d\n", client_ip, client_index);
            break;
        }
    }

    if (client_index == -1) {
        pthread_mutex_unlock(&clients_mutex);
        printf("handle_client: Released clients_mutex\n");
        char* msg = "Too many clients. Try again later.";
        printf("Connection refused for %s: %s\n", client_ip, msg);
        send(client_socket, msg, strlen(msg), 0);
        close(client_socket);
        return NULL;
    }

    ClientStatus* client = &clients[client_index];

    // Check if client is already solving a puzzle
    if (client->solving) {
        printf("Client %s is already solving a puzzle\n", client_ip);
        // Check if the client has timed out
        if (time(NULL) - client->challenge_start_time > 60) {
            printf("Client %s has timed out\n", client_ip);
            // Reset the client's status
            client->solving = false;
            mpz_clears(client->x, client->a, client->b, client->expected_r, client->expected_P, NULL);
            printf("Client %s status reset\n", client_ip);
        } else {
            pthread_mutex_unlock(&clients_mutex);
            printf("handle_client: Released clients_mutex\n");
            char* msg = "You are already solving a puzzle. Please wait.";
            printf("Puzzle request refused for %s: %s\n", client_ip, msg);
            send(client_socket, msg, strlen(msg), 0);
            close(client_socket);
            return NULL;
        }
    }

    pthread_mutex_unlock(&clients_mutex);
    printf("handle_client: Released clients_mutex\n");

    // Receive request from client
    char buffer[4096] = {0};
    int valread = read(client_socket, buffer, 4096);
    if (valread <= 0) {
        perror("read failed");
        printf("Client %s disconnected.\n", client_ip);
        close(client_socket);
        return NULL;
    }

    printf("Received from %s: %s\n", client_ip, buffer);

    if (strcmp(buffer, "REQUEST") == 0) {
        printf("Handling REQUEST from %s\n", client_ip);
        // Generate VDF parameters
        pthread_mutex_lock(&clients_mutex);
        printf("handle_client: Acquired clients_mutex\n");

        // Initialize if not already solving
        if (!client->solving) {
            printf("Initializing VDF parameters for %s\n", client_ip);
            mpz_inits(client->x, client->a, client->b, client->expected_r, client->expected_P, NULL);
        } else {
            printf("Client %s is already solving, skipping initialization\n", client_ip);
        }

        generate_vdf_parameters(client->x, client->a, client->b);

        // Send VDF puzzle to the client: "PUZZLE x a b T"
        char puzzle_msg[4096];
        gmp_sprintf(puzzle_msg, "PUZZLE %Zd %Zd %Zd %d", client->x, client->a, client->b, NUM_ITERATIONS);
        printf("Sending puzzle to %s: %s\n", client_ip, puzzle_msg);
        if (send(client_socket, puzzle_msg, strlen(puzzle_msg), 0) == -1) {
            perror("Error sending puzzle message");
        }

        // Mark client as solving
        client->solving = true;
        client->challenge_start_time = time(NULL);
        pthread_mutex_unlock(&clients_mutex);
        printf("handle_client: Released clients_mutex\n");

    } else {
        printf("Invalid request from %s\n", client_ip);
        if (send(client_socket, "Invalid request.", 16, 0) == -1) {
            perror("Error sending error message");
        }
    }

    // Wait for the solution or timeout
    time_t start_time = time(NULL);
    while (client->solving && (time(NULL) - start_time < 60)) {
        // Check for solution
        valread = recv(client_socket, buffer, sizeof(buffer), MSG_DONTWAIT); // Use non-blocking read
        if (valread > 0) {
            buffer[valread] = '\0'; // Null-terminate the received data
            if (strncmp(buffer, "SOLUTION", 8) == 0) {
                printf("Handling SOLUTION from %s\n", client_ip);
                // Client is sending a solution: "SOLUTION r P"
                mpz_t received_r, received_P;
                mpz_inits(received_r, received_P, NULL);

                char* token = strtok(buffer, " "); // Skip "SOLUTION"
                token = strtok(NULL, " ");
                if (token != NULL) {
                    mpz_set_str(received_r, token, 10);
                }
                token = strtok(NULL, " ");
                if (token != NULL) {
                    mpz_set_str(received_P, token, 10);
                }

                // Verify solution
                pthread_mutex_lock(&clients_mutex);
                printf("handle_client: Acquired clients_mutex\n");
                printf("Verifying solution from %s...\n", client_ip);
                bool valid = verify_vdf(received_r, received_P, client->a, client->b);
                if (valid) {
                    printf("Solution from %s is valid\n", client_ip);
                    // Correct solution
                    char* data = read_data();
                    if (data != NULL) {
                        printf("Sending data to %s\n", client_ip);
                        if (send(client_socket, data, strlen(data), 0) == -1) {
                            perror("Error sending data");
                        }
                        free(data);
                    } else {
                        printf("Error reading data for %s\n", client_ip);
                        if (send(client_socket, "Error reading data.", 19, 0) == -1) {
                            perror("Error sending error message");
                        }
                    }
                    printf("Client %s solved the puzzle.\n", client_ip);

                    // Reset client status
                    client->solving = false;
                    mpz_clears(client->x, client->a, client->b, client->expected_r, client->expected_P, NULL);

                } else {
                    printf("Incorrect solution from %s\n", client_ip);
                    if (send(client_socket, "Incorrect solution.", 19, 0) == -1) {
                        perror("Error sending error message");
                    }
                    printf("Client %s failed the puzzle.\n", client_ip);
                }
                pthread_mutex_unlock(&clients_mutex);
                printf("handle_client: Released clients_mutex\n");
                mpz_clears(received_r, received_P, NULL);

            }
            break; // Break out of the while loop after receiving a valid or invalid solution
        }
        else if (valread == 0)
        {
            printf("Client %s disconnected.\n", client_ip);
            break;
        }
         else {
            if (errno != EAGAIN && errno != EWOULDBLOCK) {
                perror("recv failed");
                break;
            }
        }
        sleep(1);  // Wait for 1 second before checking again
    }

    // Check if we exited the loop due to timeout
    if (client->solving) {
        printf("Timeout waiting for solution from %s\n", client_ip);
        // Reset client status
        pthread_mutex_lock(&clients_mutex);
        client->solving = false;
        mpz_clears(client->x, client->a, client->b, client->expected_r, client->expected_P, NULL);
        pthread_mutex_unlock(&clients_mutex);
    }

    close(client_socket);
    printf("Connection closed with %s\n", client_ip);
    printf("Exiting handle_client()\n");
    return NULL;
}

int main() {
    printf("Entering main()\n");
    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    // Create socket
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    // Set socket options
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    // Bind socket to port
    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    // Listen for connections
    if (listen(server_fd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    printf("Server listening on port %d\n", PORT);

    // Initialize client status array
    for (int i = 0; i < MAX_CLIENTS; i++) {
        clients[i].solving = false;
        clients[i].ip_address[0] = '\0';
    }

    // Accept incoming connections and create threads
    while (1) {
        if ((new_socket = accept(server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen)) < 0) {
            perror("accept");
            exit(EXIT_FAILURE);
        }

        pthread_t thread_id;
        if (pthread_create(&thread_id, NULL, handle_client, &new_socket) != 0) {
            perror("pthread_create failed");
            close(new_socket);
        }

        // Detach the thread so its resources are automatically released when it finishes
        pthread_detach(thread_id);
    }

    printf("Exiting main()\n");
    return 0;
}