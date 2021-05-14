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


// using 2d points
#define SIZE 10000

#define PERCEPTION_RANGE 1000
#define AVOIDANCE_RANGE 100

#define COHERENCE_RATE 0.01
#define AVOIDANCE_RATE 1.0
#define ALIGNMENT_RATE 0.125

float *newLocation;
float *newVelocity;

float *oldLocation;
float *oldVelocity;

int NCORES;

float dist2(int x1, int y1, int x2, int y2){
    return ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

void rule1(int index, float totalX, float totalY) {
    // get current distance
    // calculate average distance
    float thisX = oldLocation[index * 2];
    float thisY = oldLocation[index * 2 + 1];
    
    float averageX = (totalX - thisX) / (SIZE - 1);
    float averageY = (totalY - thisY) / (SIZE - 1);
    newVelocity[index * 2] += (averageX - thisX) * COHERENCE_RATE;
    newVelocity[index * 2 + 1] += (averageY - thisY) * COHERENCE_RATE;
}

void rule2(int index) {
    float thisX = oldLocation[index * 2];
    float thisY = oldLocation[index * 2 + 1];
    float avoidX = 0;
    float avoidY = 0;
    for (int i = 0; i < SIZE; i++){
        if (i != index){
            float thatX = oldLocation[index * 2];
            float thatY = oldLocation[index * 2 + 1];
            if (dist2(thisX, thisY, thatX, thatY) < AVOIDANCE_RANGE * AVOIDANCE_RANGE){
                avoidX -= (thatX - thisX);
                avoidY -= (thatY - thisY);
            }
        }
    }
    newVelocity[index * 2] += avoidX * AVOIDANCE_RATE;
    newVelocity[index * 2 + 1] += avoidY * AVOIDANCE_RATE;
}

void rule3(int index, float totalX, float totalY) {
    float thisX = oldVelocity[index * 2];
    float thisY = oldVelocity[index * 2 + 1];
    float averageVelX = (totalX - thisX) / (SIZE - 1);
    float averageVelY = (totalY - thisY) / (SIZE - 1);
    newVelocity[index * 2] += (averageVelX - thisX) * ALIGNMENT_RATE;
    newVelocity[index * 2 + 1] += (averageVelY - thisY) * ALIGNMENT_RATE;
}

void update() {
    float totalLocX = 0;
    float totalLocY = 0;
    float totalVelX = 0;
    float totalVelY = 0;
    #pragma omp parallel for num_threads(NCORES) reduction(+:totalLocX, totalLocY, totalVelX, totalVelY)
    for (int i = 0; i < SIZE; i++){
        totalLocX += oldLocation[i * 2];
        totalLocY += oldLocation[i * 2 + 1];
        totalVelX += oldVelocity[i * 2];
        totalVelX += oldVelocity[i * 2 + 1];
    }
    #pragma omp parallel for num_threads(NCORES)
    for (int i = 0; i < SIZE; i++){
        newVelocity[i * 2] = oldVelocity[i * 2];
        newVelocity[i * 2 + 1] = oldVelocity[i * 2 + 1];
        rule1(i, totalLocX, totalLocY);
        rule2(i);
        rule3(i, totalVelX, totalVelY);
        newLocation[i * 2] = oldLocation[i * 2] + newVelocity[i * 2];
        newLocation[i * 2 + 1] = oldLocation[i * 2 + 1] + newVelocity[i * 2 + 1];
    }
    // swap new and old location / velocity
    float *temp = oldLocation;
    oldLocation = newLocation;
    newLocation = temp;
    temp = oldVelocity;
    oldVelocity = newVelocity;
    newVelocity = temp;
}

void setup() {
    oldLocation = (float *)malloc(sizeof(float) * 2 * SIZE);
    oldVelocity = (float *)malloc(sizeof(float) * 2 * SIZE);
    newLocation = (float *)malloc(sizeof(float) * 2 * SIZE);
    newVelocity = (float *)malloc(sizeof(float) * 2 * SIZE);
    for (int i = 0; i < SIZE; i++){
        oldLocation[i * 2] = 0;
        oldLocation[i * 2 + 1] = 0;
        oldVelocity[i * 2] = 0;
        oldVelocity[i * 2 + 1] = 0;
        newLocation[i * 2] = 0;
        newLocation[i * 2 + 1] = 0;
        newVelocity[i * 2] = 0;
        newVelocity[i * 2 + 1] = 0;
    }
}

void cleanup() {
    free(oldLocation);
    free(oldVelocity);
    free(newLocation);
    free(newVelocity);
}

int main(int argc, char **argv) {
    NCORES = atoi(argv[2]);
    struct timespec before, after;
    setup();
    double average_ms = 0;
    for(int i = 0; i < 5; i++){
        clock_gettime(CLOCK_REALTIME, &before);
            update();
        clock_gettime(CLOCK_REALTIME, &after);
        double delta_ms = (double)(after.tv_sec - before.tv_sec) * 1000.0 + (after.tv_nsec - before.tv_nsec) / 1000000.0;
        average_ms += delta_ms;
        // printf("Total time: %.3lf ms\n", delta_ms);
    }
    average_ms /= 5;
    printf("Total time: %.3lf ms\n", average_ms);
    cleanup();
}