#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"

#define THREADS_PER_BLOCK_1D 256
#define THREADS_PER_BLOCK_2D 16

static vector3 *d_accels = NULL;
static double *d_mass = NULL;
static int initialized = 0;

static void checkCuda(cudaError_t result, const char *message)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", message, cudaGetErrorString(result));
        exit(1);
    }
}

static void cleanupDeviceMemory(void)
{
    if (d_hPos != NULL) {
        cudaFree(d_hPos);
        d_hPos = NULL;
    }
    if (d_hVel != NULL) {
        cudaFree(d_hVel);
        d_hVel = NULL;
    }
    if (d_mass != NULL) {
        cudaFree(d_mass);
        d_mass = NULL;
    }
    if (d_accels != NULL) {
        cudaFree(d_accels);
        d_accels = NULL;
    }
}

__global__ void computePairwiseAccelerations(vector3 *pos, double *massValues, vector3 *accels)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= NUMENTITIES || j >= NUMENTITIES) {
        return;
    }

    int index = i * NUMENTITIES + j;

    if (i == j) {
        accels[index][0] = 0.0;
        accels[index][1] = 0.0;
        accels[index][2] = 0.0;
        return;
    }

    double dx = pos[i][0] - pos[j][0];
    double dy = pos[i][1] - pos[j][1];
    double dz = pos[i][2] - pos[j][2];

    double magnitude_sq = dx * dx + dy * dy + dz * dz;
    double magnitude = sqrt(magnitude_sq);
    double accelmag = -1.0 * GRAV_CONSTANT * massValues[j] / magnitude_sq;

    accels[index][0] = accelmag * dx / magnitude;
    accels[index][1] = accelmag * dy / magnitude;
    accels[index][2] = accelmag * dz / magnitude;
}

__global__ void updateVelocitiesAndPositions(vector3 *pos, vector3 *vel, vector3 *accels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= NUMENTITIES) {
        return;
    }

    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;

    for (int j = 0; j < NUMENTITIES; j++) {
        int index = i * NUMENTITIES + j;
        ax += accels[index][0];
        ay += accels[index][1];
        az += accels[index][2];
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
    size_t vectorBytes = sizeof(vector3) * NUMENTITIES;
    size_t massBytes = sizeof(double) * NUMENTITIES;
    size_t accelBytes = sizeof(vector3) * NUMENTITIES * NUMENTITIES;

    checkCuda(cudaMalloc((void **)&d_hPos, vectorBytes), "cudaMalloc d_hPos");
    checkCuda(cudaMalloc((void **)&d_hVel, vectorBytes), "cudaMalloc d_hVel");
    checkCuda(cudaMalloc((void **)&d_mass, massBytes), "cudaMalloc d_mass");
    checkCuda(cudaMalloc((void **)&d_accels, accelBytes), "cudaMalloc d_accels");

    checkCuda(cudaMemcpy(d_hPos, hPos, vectorBytes, cudaMemcpyHostToDevice), "copy hPos to device");
    checkCuda(cudaMemcpy(d_hVel, hVel, vectorBytes, cudaMemcpyHostToDevice), "copy hVel to device");
    checkCuda(cudaMemcpy(d_mass, mass, massBytes, cudaMemcpyHostToDevice), "copy mass to device");

    atexit(cleanupDeviceMemory);
    initialized = 1;
}

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute()
{
    if (!initialized) {
        initDeviceMemory();
    }

    dim3 block2D(THREADS_PER_BLOCK_2D, THREADS_PER_BLOCK_2D);
    dim3 grid2D((NUMENTITIES + block2D.x - 1) / block2D.x,
                (NUMENTITIES + block2D.y - 1) / block2D.y);

    computePairwiseAccelerations<<<grid2D, block2D>>>(d_hPos, d_mass, d_accels);
    checkCuda(cudaGetLastError(), "launch computePairwiseAccelerations");

    int block1D = THREADS_PER_BLOCK_1D;
    int grid1D = (NUMENTITIES + block1D - 1) / block1D;

    updateVelocitiesAndPositions<<<grid1D, block1D>>>(d_hPos, d_hVel, d_accels);
    checkCuda(cudaGetLastError(), "launch updateVelocitiesAndPositions");
    checkCuda(cudaDeviceSynchronize(), "synchronize after kernels");

    checkCuda(cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost), "copy hPos to host");
    checkCuda(cudaMemcpy(hVel, d_hVel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDeviceToHost), "copy hVel to host");
}