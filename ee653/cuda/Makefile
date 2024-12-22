# compiler and flags

NVCC = /usr/local/cuda-12.6/bin/nvcc
CFLAGS = -arch=sm_75 -O2

# Target binary name
TARGET = hello_cuda
XGCD = xgcd

# Source file
SRC = hello_cuda.cu
XGC = xgcd.cu

# Build rule
all: $(TARGET) $(XGCD)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRC)

$(XGCD): $(XGC)
	$(NVCC) $(CFLAGS) -o $(XGCD) $(XGC)

run: $(TARGET) $(XGCD)
	./$(XGCD) 
	./$(TARGET)

clean:
	rm -f $(TARGET) $(XGCD)

