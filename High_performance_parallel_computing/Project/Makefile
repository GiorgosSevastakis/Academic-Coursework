# Makefile for compiling the CUDA Ising model on RTX 3050 Ti (compute capability 8.6). 
# Change depending on the GPU!

NVCC = nvcc
TARGET = kernel
SRC = kernel.cu

# Set architecture for Ampere (RTX 3050 Ti is sm_86)
ARCH = -arch=sm_86

# Output directory for frames
FRAMES_DIR = frames

# Compiler flags
CFLAGS = -O3

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(ARCH) $(CFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
	rm -f $(FRAMES_DIR)/*.pgm
