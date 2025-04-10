# Compiler options
NVCC = nvcc
CXX = g++
NVCC_FLAGS = -O3 -arch=sm_61
CXX_FLAGS = -O3 -std=c++11

# Libraries
LIBS = -lGL -lGLEW -lglfw -lcudart

# Source files
SRC = nbody_simulation.cu

# Output executable
EXEC = nbody_sim

all: $(EXEC)

$(EXEC): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LIBS)

clean:
	rm -f $(EXEC)
