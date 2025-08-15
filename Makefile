# Makefile for building CUDA binary

# Compiler
NVCC = nvcc

# Architecture
ARCH = -arch=sm_86

# Flags
NVCC_FLAGS = $(ARCH) --expt-relaxed-constexpr

# Source and output
SRC = main.cu
OUT = main

# Default target
all: $(OUT)

# Build rule
$(OUT): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Clean up
clean:
	rm -f $(OUT)

