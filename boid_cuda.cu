#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <error.h>
#include <limits.h>
#include <pthread.h>
#include <time.h>


#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "boid_rules.h"

int SIZE = 10000;

#define PERCEPTION_RANGE 1000
#define AVOIDANCE_RANGE 100

#define COHERENCE_RATE 0.01
#define AVOIDANCE_RATE 1.0
#define ALIGNMENT_RATE 0.125

#define SCAN_BLOCK_DIM 256

#define DEBUG false

float *newLocation;
float *newVelocity;

float *oldLocation;
float *oldVelocity;

struct cudaDeviceInfo {
    float* oldLocation;
    float* oldVelocity;
    float* newLocation;
    float* newVelocity;
};

__device__ cudaDeviceInfo cudaData;

__device__ float deviceDist2(int x1, int y1, int x2, int y2){
    return ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

__global__ void kernelRule1(cudaDeviceInfo cudaData, int SIZE) {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int localId = threadIdx.x;
    int workPerCycle = SCAN_BLOCK_DIM;
    int workCycles = (SIZE + SCAN_BLOCK_DIM - 1) / SCAN_BLOCK_DIM;
    float thisX, thisY;
    if (globalId < SIZE){
        thisX =  cudaData.oldLocation[globalId * 2];
        thisY = cudaData.oldLocation[globalId * 2 + 1];
    }
    float averageX = 0;
    float averageY = 0;
    int count = 0;

    __shared__ float2 sharedLocation[SCAN_BLOCK_DIM];
    for (int i = 0; i < workCycles; i++){
        int loadId = localId + i * workPerCycle;
        if (loadId < SIZE){
            sharedLocation[localId] = *(float2 *)&cudaData.oldLocation[loadId * 2];
        }
        __syncthreads();
        if (globalId < SIZE){
            for (int j = 0; j < workPerCycle; j++){
                if (globalId != j + workPerCycle * i && j + workPerCycle * i < SIZE){
                    float thatX = ((float *)sharedLocation)[j * 2];
                    float thatY = ((float *)sharedLocation)[j * 2 + 1];
                    if (deviceDist2(thisX, thisY, thatX, thatY) < PERCEPTION_RANGE * PERCEPTION_RANGE){
                        averageX += thatX;
                        averageY += thatY;
                        count += 1;
                    }
                }
            }
        }
    }
    averageX /= count;
    averageY /= count;
    if (globalId < SIZE){
        cudaData.newVelocity[globalId * 2] += (averageX - thisX) * COHERENCE_RATE;
        cudaData.newVelocity[globalId * 2 + 1] += (averageY - thisY) * COHERENCE_RATE;
    }
}

__global__ void kernelRule2(cudaDeviceInfo cudaData, int SIZE) {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int localId = threadIdx.x;
    int workPerCycle = SCAN_BLOCK_DIM;
    int workCycles = (SIZE + SCAN_BLOCK_DIM - 1) / SCAN_BLOCK_DIM;
    float thisX, thisY;
    if (globalId < SIZE){
        thisX =  cudaData.oldLocation[globalId * 2];
        thisY = cudaData.oldLocation[globalId * 2 + 1];
    }
    float avoidX = 0;
    float avoidY = 0;

    __shared__ float2 sharedLocation[SCAN_BLOCK_DIM];
    for (int i = 0; i < workCycles; i++){
        int loadId = localId + i * workPerCycle;
        if (loadId < SIZE){
            sharedLocation[localId] = *(float2 *)&cudaData.oldLocation[loadId * 2];
        }
        __syncthreads();
        if (globalId < SIZE){
            for (int j = 0; j < workPerCycle; j++){
                if (globalId != j + workPerCycle * i && j + workPerCycle * i < SIZE){
                    float thatX = ((float *)sharedLocation)[j * 2];
                    float thatY = ((float *)sharedLocation)[j * 2 + 1];
                    if (deviceDist2(thisX, thisY, thatX, thatY) < AVOIDANCE_RANGE * AVOIDANCE_RANGE){
                        avoidX -= (thatX - thisX);
                        avoidY -= (thatY - thisY);
                    }
                }
            }
        }
    }
    if (globalId < SIZE){
        cudaData.newVelocity[globalId * 2] += avoidX * AVOIDANCE_RATE;
        cudaData.newVelocity[globalId * 2 + 1] += avoidY * AVOIDANCE_RATE;
    }
}

__global__ void kernelRule3(cudaDeviceInfo cudaData, int SIZE) {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int localId = threadIdx.x;
    int workPerCycle = SCAN_BLOCK_DIM;
    int workCycles = (SIZE + SCAN_BLOCK_DIM - 1) / SCAN_BLOCK_DIM;
    float thisX, thisY;
    if (globalId < SIZE){
        thisX =  cudaData.oldLocation[globalId * 2];
        thisY = cudaData.oldLocation[globalId * 2 + 1];
    }
    float averageVelX = 0;
    float averageVelY = 0;
    int count = 0;

    __shared__ float2 sharedLocation[SCAN_BLOCK_DIM];
    __shared__ float2 sharedVelocity[SCAN_BLOCK_DIM];
    for (int i = 0; i < workCycles; i++){
        int loadId = localId + i * workPerCycle;
        if (loadId < SIZE){
            sharedLocation[localId] = *(float2 *)&cudaData.oldLocation[loadId * 2];
            sharedVelocity[localId] = *(float2 *)&cudaData.oldVelocity[loadId * 2];
        }
        __syncthreads();
        if (globalId < SIZE){
            for (int j = 0; j < workPerCycle; j++){
                if (globalId != j + workPerCycle * i && j + workPerCycle * i < SIZE){
                    float thatX = ((float *)sharedLocation)[j * 2];
                    float thatY = ((float *)sharedLocation)[j * 2 + 1];
                    if (deviceDist2(thisX, thisY, thatX, thatY) < AVOIDANCE_RANGE * AVOIDANCE_RANGE){
                        averageVelX += ((float *)sharedVelocity)[j * 2];
                        averageVelY += ((float *)sharedVelocity)[j * 2 + 1];
                        count ++;
                    }
                }
            }
        }
    }
    averageVelX /= count;
    averageVelY /= count;
    if (globalId < SIZE){
        cudaData.newVelocity[globalId * 2] += (averageVelX - cudaData.oldVelocity[globalId * 2]) * ALIGNMENT_RATE;
        cudaData.newVelocity[globalId * 2 + 1] += (averageVelY - cudaData.oldVelocity[globalId * 2 + 1]) * ALIGNMENT_RATE;
    }
}

__global__ void kernelUpdateLoc(cudaDeviceInfo cudaData, int SIZE) {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalId < SIZE){
        cudaData.newLocation[globalId * 2] += cudaData.newVelocity[globalId * 2];
        cudaData.newLocation[globalId * 2 + 1] += cudaData.newVelocity[globalId * 2 + 1];
    }
}

__global__ void kernelNew2Old(cudaDeviceInfo cudaData, int SIZE) {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalId < SIZE){
        cudaData.oldLocation[globalId * 2] = cudaData.newLocation[globalId * 2];
        cudaData.oldLocation[globalId * 2 + 1] = cudaData.newLocation[globalId * 2 + 1];
        cudaData.oldVelocity[globalId * 2] = cudaData.newVelocity[globalId * 2];
        cudaData.oldVelocity[globalId * 2 + 1] = cudaData.newVelocity[globalId * 2 + 1];
    }
}

void update() {
    int numBlocks = (SIZE + SCAN_BLOCK_DIM - 1) / SCAN_BLOCK_DIM;
    int blockSize = SCAN_BLOCK_DIM;
    kernelRule1<<<numBlocks, blockSize>>>(cudaData, SIZE);
    cudaDeviceSynchronize();
    if (DEBUG) printf("Error at rule 1: %s\n", cudaGetErrorString(cudaGetLastError()));
    kernelRule2<<<numBlocks, blockSize>>>(cudaData, SIZE);
    cudaDeviceSynchronize();
    if (DEBUG) printf("Error at rule 2: %s\n", cudaGetErrorString(cudaGetLastError()));
    kernelRule3<<<numBlocks, blockSize>>>(cudaData, SIZE);
    cudaDeviceSynchronize();
    if (DEBUG) printf("Error at rule 3: %s\n", cudaGetErrorString(cudaGetLastError()));
    kernelUpdateLoc<<<numBlocks, blockSize>>>(cudaData, SIZE);
    cudaDeviceSynchronize();
    if (DEBUG) printf("Error at update loc: %s\n", cudaGetErrorString(cudaGetLastError()));
    kernelNew2Old<<<numBlocks, blockSize>>>(cudaData, SIZE);
    cudaDeviceSynchronize();
    if (DEBUG) printf("Error at copy loc/vel: %s\n", cudaGetErrorString(cudaGetLastError()));
}
void updateLinear() {
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

void setupCuda() {
    // declare device memory and copy data to device memory
    float *cudaDeviceOldLocation;
    float *cudaDeviceOldVelocity;
    float *cudaDeviceNewLocation;
    float *cudaDeviceNewVelocity;
    cudaMalloc(&cudaDeviceOldLocation, sizeof(float) * 2 * SIZE);
    cudaMalloc(&cudaDeviceNewLocation, sizeof(float) * 2 * SIZE);
    cudaMalloc(&cudaDeviceOldVelocity, sizeof(float) * 2 * SIZE);
    cudaMalloc(&cudaDeviceNewVelocity, sizeof(float) * 2 * SIZE);
    cudaMemcpy(cudaDeviceOldLocation, oldLocation, sizeof(float) * 2 * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceNewLocation, newLocation, sizeof(float) * 2 * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceOldVelocity, oldVelocity, sizeof(float) * 2 * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceNewVelocity, newVelocity, sizeof(float) * 2 * SIZE, cudaMemcpyHostToDevice);
    cudaData.oldLocation = cudaDeviceOldLocation;
    cudaData.newLocation = cudaDeviceNewLocation;
    cudaData.oldVelocity = cudaDeviceOldVelocity;
    cudaData.newVelocity = cudaDeviceNewVelocity;
    if (DEBUG) printf("Error at setup: %s\n", cudaGetErrorString(cudaGetLastError()));
}

void cleanup() {
    free(oldLocation);
    free(oldVelocity);
    free(newLocation);
    free(newVelocity);
    cudaFree(cudaData.oldLocation);
    cudaFree(cudaData.newLocation);
    cudaFree(cudaData.oldVelocity);
    cudaFree(cudaData.newVelocity);
}

int main(int argc, char **argv) {
    // NCORES = atoi(argv[2]);
    int iterations = 1;
    if (argc >= 3)
        iterations = atoi(argv[2]);
    if (argc >= 5)
        SIZE = atoi(argv[4]);
    struct timespec before, after;
    oldLocation = (float *)malloc(sizeof(float) * 2 * SIZE);
    oldVelocity = (float *)malloc(sizeof(float) * 2 * SIZE);
    newLocation = (float *)malloc(sizeof(float) * 2 * SIZE);
    newVelocity = (float *)malloc(sizeof(float) * 2 * SIZE);
    setup(oldLocation, newLocation, oldVelocity, newVelocity, SIZE);
    setupCuda();
    double average_ms = 0;
    for(int i = 0; i < iterations; i++){
        clock_gettime(CLOCK_REALTIME, &before);
            update();
        clock_gettime(CLOCK_REALTIME, &after);
        double delta_ms = (double)(after.tv_sec - before.tv_sec) * 1000.0 + (after.tv_nsec - before.tv_nsec) / 1000000.0;
        average_ms += delta_ms;
    }
    average_ms /= iterations;
    double cuda_time = average_ms;
    printf("Total time with cuda: %.3lf ms\n", average_ms);
    cleanup();

    oldLocation = (float *)malloc(sizeof(float) * 2 * SIZE);
    oldVelocity = (float *)malloc(sizeof(float) * 2 * SIZE);
    newLocation = (float *)malloc(sizeof(float) * 2 * SIZE);
    newVelocity = (float *)malloc(sizeof(float) * 2 * SIZE);
    setup(oldLocation, newLocation, oldVelocity, newVelocity, SIZE);
    setupCuda();
    average_ms = 0;
    for(int i = 0; i < iterations; i++){
        clock_gettime(CLOCK_REALTIME, &before);
            updateLinear();
        clock_gettime(CLOCK_REALTIME, &after);
        double delta_ms = (double)(after.tv_sec - before.tv_sec) * 1000.0 + (after.tv_nsec - before.tv_nsec) / 1000000.0;
        average_ms += delta_ms;
    }
    average_ms /= iterations;
    double normal_time = average_ms;
    printf("Total time linear: %.3lf ms\n", average_ms);
    printf("Speedup: %0.3f\n", normal_time / cuda_time);
    cleanup();
}