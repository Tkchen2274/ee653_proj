CC = gcc
CFLAGS = -Wall -g -lgmp -pthread

all: client

client: client.c
	$(CC) $(CFLAGS) client.c -o client

.PHONY: clean
clean:
	rm -f client
