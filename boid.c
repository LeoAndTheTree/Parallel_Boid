#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <error.h>
#include <limits.h>
#include <pthread.h>
#include <time.h>

#include <omp.h>

#include "boid_rules.h"


int SIZE = 10000;

float *newLocation;
float *newVelocity;

float *oldLocation;
float *oldVelocity;

int NCORES = 1;


void update() {
    #pragma omp parallel for num_threads(NCORES)
    for (int i = 0; i < SIZE; i++){
        rule1(i, oldLocation, newLocation, oldVelocity, newVelocity, SIZE);
        rule2(i, oldLocation, newLocation, oldVelocity, newVelocity, SIZE);
        rule3(i, oldLocation, newLocation, oldVelocity, newVelocity, SIZE);
        newLocation[i * 2] = oldLocation[i * 2] + newVelocity[i * 2];
        newLocation[i * 2 + 1] = oldLocation[i * 2 + 1] + newVelocity[i * 2 + 1];
    }
    memcpy(oldLocation, newLocation, sizeof(float) * 2 * SIZE);
    memcpy(oldVelocity, newVelocity, sizeof(float) * 2 * SIZE);
}

void cleanup() {
    free(oldLocation);
    free(oldVelocity);
    free(newLocation);
    free(newVelocity);
}

int main(int argc, char **argv) {
    if (argc >= 3)
        NCORES = atoi(argv[2]);
    if (argc >= 5)
        SIZE = atoi(argv[4]);
    struct timespec before, after;
    oldLocation = (float *)malloc(sizeof(float) * 2 * SIZE);
    oldVelocity = (float *)malloc(sizeof(float) * 2 * SIZE);
    newLocation = (float *)malloc(sizeof(float) * 2 * SIZE);
    newVelocity = (float *)malloc(sizeof(float) * 2 * SIZE);
    setup(oldLocation, newLocation, oldVelocity, newVelocity, SIZE);
    double average_ms = 0;
    for(int i = 0; i < 5; i++){
        clock_gettime(CLOCK_REALTIME, &before);
            update();
        clock_gettime(CLOCK_REALTIME, &after);
        double delta_ms = (double)(after.tv_sec - before.tv_sec) * 1000.0 + (after.tv_nsec - before.tv_nsec) / 1000000.0;
        average_ms += delta_ms;
    }
    average_ms /= 5;
    printf("Total time: %.3lf ms\n", average_ms);
    cleanup();
}