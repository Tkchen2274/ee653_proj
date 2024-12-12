CC = gcc

all: serial openmp openmp2 openmp3 serial_alt
 
serial: serial.c
	$(CC) serial.c -o serial

serial_alt: serial_alt.c
	$(CC) serial_alt.c -o serial_alt

openmp: openmp.c
	$(CC) -fopenmp -o openmp openmp.c

openmp2: openmp2.c
	$(CC) -fopenmp -o openmp2 openmp2.c

openmp3: openmp3.c
	$(CC) -fopenmp -o openmp3 openmp3.c


clean:
	rm serial openmp openmp2 openmp3
