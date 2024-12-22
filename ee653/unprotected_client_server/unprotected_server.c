#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>


#define PORT 8080
#define MAX_CLIENTS 10

// Client status structure (no longer needed for VDF)
typedef struct {
    char ip_address[INET_ADDRSTRLEN];
    // No longer need VDF related fields
} ClientStatus;

ClientStatus clients[MAX_CLIENTS];
// No longer need the mutex for client status
// pthread_mutex_t clients_mutex = PTHREAD_MUTEX_INITIALIZER;

// Function to read data from data.txt
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
        // Directly read and send the data
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
    } else {
        printf("Invalid request from %s\n", client_ip);
        if (send(client_socket, "Invalid request.", 16, 0) == -1) {
            perror("Error sending error message");
        }
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

    // Initialize client status array (not strictly necessary now, but kept for potential future use)
    for (int i = 0; i < MAX_CLIENTS; i++) {
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
