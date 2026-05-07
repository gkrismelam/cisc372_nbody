#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "vector.h"
#include "config.h"

static vector3 *d_hPos = NULL;
static vector3 *d_hVel = NULL;
static double  *d_mass = NULL;
static vector3 *d_accels = NULL;

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
        accels[idx][0] = 0.0;
        accels[idx][1] = 0.0;
        accels[idx][2] = 0.0;
        return;
    }

    double dx = pos[i][0] - pos[j][0];
    double dy = pos[i][1] - pos[j][1];
    double dz = pos[i][2] - pos[j][2];

    double r2 = dx*dx + dy*dy + dz*dz;

    if (r2 == 0.0) {
        accels[idx][0] = accels[idx][1] = accels[idx][2] = 0.0;
        return;
    }

    double r = sqrt(r2);
    double accelmag = -GRAV_CONSTANT * massValues[j] / r2;

    accels[idx][0] = accelmag * dx / r;
    accels[idx][1] = accelmag * dy / r;
    accels[idx][2] = accelmag * dz / r;
}

__global__ void updateVelocitiesAndPositions(
    vector3 *pos, vector3 *vel, vector3 *accels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= NUMENTITIES) return;

    double ax = 0.0, ay = 0.0, az = 0.0;

    for (int j = 0; j < NUMENTITIES; j++) {
        int idx = i * NUMENTITIES + j;
        ax += accels[idx][0];
        ay += accels[idx][1];
        az += accels[idx][2];
    }

    vel[i][0] += ax * INTERVAL;
    vel[i][1] += ay * INTERVAL;
    vel[i][2] += az * INTERVAL;

    pos[i][0] += vel[i][0] * INTERVAL;
    pos[i][1] += vel[i][1] * INTERVAL;
    pos[i][2] += vel[i][2] * INTERVAL;
}

static void initDeviceMemory(void)
{
    size_t vecBytes = sizeof(vector3) * NUMENTITIES;
    size_t massBytes = sizeof(double) * NUMENTITIES;
    size_t accelBytes = sizeof(vector3) * NUMENTITIES * NUMENTITIES;

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

void compute()
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