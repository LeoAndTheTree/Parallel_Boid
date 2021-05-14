#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <error.h>
#include <limits.h>
#include <pthread.h>
#include <time.h>

#include <mpi.h>

#include "boid_rules.h"


// using 2d points
int SIZE = 10000;

#define PERCEPTION_RANGE 1000
#define AVOIDANCE_RANGE 100

#define COHERENCE_RATE 0.01
#define AVOIDANCE_RATE 1.0
#define ALIGNMENT_RATE 0.125

void update(float *oldLocation, float*newLocation, float *oldVelocity, float *newVelocity, int SIZE) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // only do work within your rank.
    int workUnit = (SIZE + size - 1) / size;
    int workStart = workUnit * rank;
    int workEnd = workUnit * (rank + 1);
    if (workEnd > SIZE)
        workEnd = SIZE;
    for (int i = workStart; i < workEnd; i++){
        rule1(i, oldLocation, newLocation, oldVelocity, newVelocity, SIZE);
        rule2(i, oldLocation, newLocation, oldVelocity, newVelocity, SIZE);
        rule3(i, oldLocation, newLocation, oldVelocity, newVelocity, SIZE);
        newLocation[i * 2] = oldLocation[i * 2] + newVelocity[i * 2];
        newLocation[i * 2 + 1] = oldLocation[i * 2 + 1] + newVelocity[i * 2 + 1];
    }
    memcpy(&oldLocation[rank * workUnit * 2], &newLocation[rank * workUnit * 2], sizeof(float) * 2 * (workEnd - workStart));
    memcpy(&oldVelocity[rank * workUnit * 2], &newVelocity[rank * workUnit * 2], sizeof(float) * 2 * (workEnd - workStart));
    // send location to all other place
    for (int i = 0; i < size; i++){
        MPI_Bcast(&newLocation[i * workUnit * 2], workUnit * 2, MPI_FLOAT, i, MPI_COMM_WORLD);
        MPI_Bcast(&newVelocity[i * workUnit * 2], workUnit * 2, MPI_FLOAT, i, MPI_COMM_WORLD);
        MPI_Bcast(&oldLocation[i * workUnit * 2], workUnit * 2, MPI_FLOAT, i, MPI_COMM_WORLD);
        MPI_Bcast(&oldVelocity[i * workUnit * 2], workUnit * 2, MPI_FLOAT, i, MPI_COMM_WORLD);
    }
}

int main(int argc, char **argv) {
    if (argc >= 3)
        SIZE = atoi(argv[2]);
    MPI_Init(NULL, NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    float *oldLocation = (float *)malloc(sizeof(float) * 2 * SIZE);
    float *oldVelocity = (float *)malloc(sizeof(float) * 2 * SIZE);
    float *newLocation = (float *)malloc(sizeof(float) * 2 * SIZE);
    float *newVelocity = (float *)malloc(sizeof(float) * 2 * SIZE);
    setup(oldLocation, newLocation, oldVelocity, newVelocity, SIZE);
    double start_time = MPI_Wtime();
    for(int i = 0; i < 5; i++){
        update(oldLocation, newLocation, oldVelocity, newVelocity, SIZE);
    }
    double end_time = MPI_Wtime();
    if (rank == 0){
        putchar('\n');
        printf("============ Time ============\n");
        printf("Time: %.3f ms (%.3f s)\n", (end_time - start_time) * 1000, end_time - start_time);
    }

    free(oldLocation);
    free(oldVelocity);
    free(newLocation);
    free(newVelocity);

    MPI_Finalize();
}