#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 100000000  // vector size

// Simple CUDA error-check macro
#define CUDA_CHECK(x) do { \
    cudaError_t err = x; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s (at %s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

__global__ void vecAdd(float *a, float *b, float *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) c[i] = a[i] + b[i];
}

int main() {
    float *a, *b, *c_cpu, *c_gpu;
    float *d_a, *d_b, *d_c;

    a = (float*)malloc(N*sizeof(float));
    b = (float*)malloc(N*sizeof(float));
    c_cpu = (float*)malloc(N*sizeof(float));
    c_gpu = (float*)malloc(N*sizeof(float));

    for(int i = 0; i < N; i++){
        a[i] = i;
        b[i] = N - i;
    }

    // CPU timing
    clock_t t1 = clock();
    for(int i = 0; i < N; i++)
        c_cpu[i] = a[i] + b[i];
    clock_t t2 = clock();
    double cpu_time = (double)(t2 - t1) / CLOCKS_PER_SEC;

    // GPU memory allocation
    CUDA_CHECK(cudaMalloc(&d_a, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, a, N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, N*sizeof(float), cudaMemcpyHostToDevice));

    // GPU timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int block = 256;
    int grid = (N + block - 1) / block;

    CUDA_CHECK(cudaEventRecord(start));
    vecAdd<<<grid, block>>>(d_a, d_b, d_c);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaMemcpy(c_gpu, d_c, N*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));

    // Output
    printf("CPU time: %f seconds\n", cpu_time);
    printf("GPU time: %f milliseconds\n", gpu_time);

    // Cleanup
    free(a); free(b); free(c_cpu); free(c_gpu);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}

