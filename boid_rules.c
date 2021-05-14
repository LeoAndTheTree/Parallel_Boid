#include "boid_rules.h"

#define PERCEPTION_RANGE 1000
#define AVOIDANCE_RANGE 100

#define COHERENCE_RATE 0.01
#define AVOIDANCE_RATE 1.0
#define ALIGNMENT_RATE 0.125

float dist2(int x1, int y1, int x2, int y2){
    return ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

void rule1(int index, float *oldLocation, float *newLocation, float *oldVelocity, float *newVelocity, int SIZE) {
    // get current distance
    // calculate average distance
    float thisX = oldLocation[index * 2];
    float thisY = oldLocation[index * 2 + 1];
    float averageX = 0;
    float averageY = 0;
    int count = 0;
    // #pragma omp parallel for num_threads(NCORES) reduction(+:averageX,averageY,count)
        for (int i = 0; i < SIZE; i++){
            if (i != index){
                // printf("Index: %d, i: %d\n", index, i);
                float thatX = oldLocation[i * 2];
                float thatY = oldLocation[i * 2 + 1];
                if (dist2(thisX, thisY, thatX, thatY) < PERCEPTION_RANGE * PERCEPTION_RANGE){
                    averageX += thatX;
                    averageY += thatY;
                    count += 1;
                }
            }
        }
    
    averageX /= count;
    averageY /= count;
    // printf("Index: %d, averageX: %f, averageY: %f\n", index, averageX, averageY);
    newVelocity[index * 2] += (averageX - thisX) * COHERENCE_RATE;
    newVelocity[index * 2 + 1] += (averageY - thisY) * COHERENCE_RATE;
}

void rule2(int index, float *oldLocation, float *newLocation, float *oldVelocity, float *newVelocity, int SIZE) {
    float thisX = oldLocation[index * 2];
    float thisY = oldLocation[index * 2 + 1];
    float avoidX = 0;
    float avoidY = 0;
    for (int i = 0; i < SIZE; i++){
        if (i != index){
            float thatX = oldLocation[i * 2];
            float thatY = oldLocation[i * 2 + 1];
            if (dist2(thisX, thisY, thatX, thatY) < AVOIDANCE_RANGE * AVOIDANCE_RANGE){
                avoidX -= (thatX - thisX);
                avoidY -= (thatY - thisY);
            }
        }
    }
    newVelocity[index * 2] += avoidX * AVOIDANCE_RATE;
    newVelocity[index * 2 + 1] += avoidY * AVOIDANCE_RATE;
}

void rule3(int index, float *oldLocation, float *newLocation, float *oldVelocity, float *newVelocity, int SIZE) {
    float thisX = oldLocation[index * 2];
    float thisY = oldLocation[index * 2 + 1];
    float averageVelX = 0;
    float averageVelY = 0;
    int count = 0;
    for (int i = 0; i < SIZE; i++){
        if (i != index){
            float thatX = oldLocation[i * 2];
            float thatY = oldLocation[i * 2 + 1];
            if (dist2(thisX, thisY, thatX, thatY) < PERCEPTION_RANGE * PERCEPTION_RANGE){
                averageVelX += oldVelocity[i * 2];
                averageVelY += oldVelocity[i * 2 + 1];
                count ++;
            }
        }
    }
    averageVelX /= count;
    averageVelY /= count;
    newVelocity[index * 2] += (averageVelX - oldVelocity[index * 2]) * ALIGNMENT_RATE;
    newVelocity[index * 2 + 1] += (averageVelY - oldVelocity[index * 2 + 1]) * ALIGNMENT_RATE;
}

void setup(float *oldLocation, float *newLocation, float *oldVelocity, float *newVelocity, int SIZE) {
    for (int i = 0; i < SIZE; i++){
        oldLocation[i * 2] = i;
        oldLocation[i * 2 + 1] = i;
        oldVelocity[i * 2] = 0;
        oldVelocity[i * 2 + 1] = 0;
        newLocation[i * 2] = i;
        newLocation[i * 2 + 1] = i;
        newVelocity[i * 2] = 0;
        newVelocity[i * 2 + 1] = 0;
    }
}