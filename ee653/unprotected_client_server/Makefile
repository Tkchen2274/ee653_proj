CC = gcc
CFLAGS = -pthread

all: unprotected_server

server: server.c
	$(CC) unprotected_server.c -o unprotected_server $(CFLAGS) 

.PHONY: clean
clean:
	rm -f unprotected_server
