#ifndef __BOID_RULES_H__
#define __BOID_RULES_H__

void rule1(int index, float *oldLocation, float *newLocation, float *oldVelocity, float *newVelocity, int SIZE);
void rule2(int index, float *oldLocation, float *newLocation, float *oldVelocity, float *newVelocity, int SIZE);
void rule3(int index, float *oldLocation, float *newLocation, float *oldVelocity, float *newVelocity, int SIZE);
void setup(float *oldLocation, float *newLocation, float *oldVelocity, float *newVelocity, int SIZE);

#endif
