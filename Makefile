CXXFLAGS += -Wall -Wextra -pthread -fopenmp -g -O3 -DNDEBUG

.phony: all boid release

all: release


release: boid.c boid_inf.c boid_mpi.c boid_cuda.cu boid_rules.c
	g++ boid.c boid_rules.c -o boid $(CXXFLAGS) -w
	mpicxx boid_mpi.c boid_rules.c -o boid_mpi $(CXXFLAGS) -w
	g++ boid_rules.c -c -o boid_rules.o $(CXXFLAGS) -w
	nvcc -O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc boid_cuda.cu -c -o boid_cuda.o -w
	g++ boid_rules.o boid_cuda.o -o boid_cuda -L/usr/local/depot/cuda-10.2/lib64/ -lcudart

clean:
	rm -f ./boid
	rm -f ./boid_inf
