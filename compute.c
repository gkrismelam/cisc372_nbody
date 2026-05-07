#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "vector.h"
#include "config.h"

// Removed 'static' keyword to match extern declarations in vector.h
vector3 *d_hPos = NULL;
vector3 *d_hVel = NULL;
double  *d_mass = NULL;
vector3 *d_accels = NULL;

static int initialized = 0;

static void checkCuda(cudaError_t result, const char *message)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n",
                message, cudaGetErrorString(result));
        exit(1);
    }
}

static void cleanupDeviceMemory(void)
{
    if (d_hPos) cudaFree(d_hPos);
    if (d_hVel) cudaFree(d_hVel);
    if (d_mass) cudaFree(d_mass);
    if (d_accels) cudaFree(d_accels);
}

__global__ void computePairwiseAccelerations(
    vector3 *pos, double *massValues, vector3 *accels)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= NUMENTITIES || j >= NUMENTITIES) return;

    int idx = i * NUMENTITIES + j;

    if (i == j) {
        accels[idx].x = 0.0;
        accels[idx].y = 0.0;
        accels[idx].z = 0.0;
        return;
    }

    double dx = pos[i].x - pos[j].x;
    double dy = pos[i].y - pos[j].y;
    double dz = pos[i].z - pos[j].z;

    double r2 = dx*dx + dy*dy + dz*dz;

    if (r2 == 0.0) {
        accels[idx].x = accels[idx].y = accels[idx].z = 0.0;
        return;
    }

    double r = sqrt(r2);
    double accelmag = -GRAV_CONSTANT * massValues[j] / r2;

    accels[idx].x = accelmag * dx / r;
    accels[idx].y = accelmag * dy / r;
    accels[idx].z = accelmag * dz / r;
}

__global__ void updateVelocitiesAndPositions(
    vector3 *pos, vector3 *vel, vector3 *accels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= NUMENTITIES) return;

    double ax = 0.0, ay = 0.0, az = 0.0;

    for (int j = 0; j < NUMENTITIES; j++) {
        int idx = i * NUMENTITIES + j;
        ax += accels[idx].x;
        ay += accels[idx].y;
        az += accels[idx].z;
    }

    vel[i].x += ax * INTERVAL;
    vel[i].y += ay * INTERVAL;
    vel[i].z += az * INTERVAL;

    pos[i].x += vel[i].x * INTERVAL;
    pos[i].y += vel[i].y * INTERVAL;
    pos[i].z += vel[i].z * INTERVAL;
}

static void initDeviceMemory(void)
{
    size_t vecBytes = sizeof(vector3) * NUMENTITIES;
    size_t massBytes = sizeof(double) * NUMENTITIES;
    size_t accelBytes = sizeof(vector3) * NUMENTITIES * NUMENTITIES;

    // Note: hPos, hVel, and mass need to be defined in main.c or another file
    checkCuda(cudaMalloc(&d_hPos, vecBytes), "malloc pos");
    checkCuda(cudaMalloc(&d_hVel, vecBytes), "malloc vel");
    checkCuda(cudaMalloc(&d_mass, massBytes), "malloc mass");
    checkCuda(cudaMalloc(&d_accels, accelBytes), "malloc accels");

    checkCuda(cudaMemcpy(d_hPos, hPos, vecBytes, cudaMemcpyHostToDevice), "copy pos");
    checkCuda(cudaMemcpy(d_hVel, hVel, vecBytes, cudaMemcpyHostToDevice), "copy vel");
    checkCuda(cudaMemcpy(d_mass, mass, massBytes, cudaMemcpyHostToDevice), "copy mass");

    atexit(cleanupDeviceMemory);
    initialized = 1;
}

void compute(void)
{
    if (!initialized) {
        initDeviceMemory();
    }

    dim3 block2D(16, 16);
    dim3 grid2D(
        (NUMENTITIES + 15) / 16,
        (NUMENTITIES + 15) / 16
    );

    computePairwiseAccelerations<<<grid2D, block2D>>>(
        d_hPos, d_mass, d_accels);

    checkCuda(cudaGetLastError(), "kernel 1");

    int block1D = 256;
    int grid1D = (NUMENTITIES + block1D - 1) / block1D;

    updateVelocitiesAndPositions<<<grid1D, block1D>>>(
        d_hPos, d_hVel, d_accels);

    checkCuda(cudaGetLastError(), "kernel 2");
    checkCuda(cudaDeviceSynchronize(), "sync");

    checkCuda(cudaMemcpy(hPos, d_hPos,
        sizeof(vector3) * NUMENTITIES,
        cudaMemcpyDeviceToHost), "copy back pos");

    checkCuda(cudaMemcpy(hVel, d_hVel,
        sizeof(vector3) * NUMENTITIES,
        cudaMemcpyDeviceToHost), "copy back vel");
}