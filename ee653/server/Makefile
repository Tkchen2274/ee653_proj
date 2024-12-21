CC = gcc
CFLAGS = -Wall -g -pthread
LDFLAGS = -lgmp

all: server

server: server.c
	$(CC) $(CFLAGS) server.c -o server $(LDFLAGS)

.PHONY: clean
clean:
	rm -f server
