#include <stdio.h>
#include <cuda.h>
__global__ void demoKernel() {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float x = idx * 0.1f;
// Use the variable to avoid warning
if (x < 0) printf("");
}
int main() {
// Print GPU properties
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("=== GPU Device Properties ===\n");
printf("Device Name: %s\n", prop.name);
printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
printf("Multiprocessor Count: %d\n\n", prop.multiProcessorCount);
int blocks = 4;
int threads = 256;
printf("Launching kernel <<<%d blocks, %d threads>>>\n\n", blocks, threads);
// Timing using CUDA events
cudaEvent_t start, stop;
float time_ms = 0.0f;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
demoKernel<<<blocks, threads>>>();
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&time_ms, start, stop);
printf("Execution Time: %.6f ms\n", time_ms);
cudaEventDestroy(start);
 cudaEventDestroy(stop);
return 0;
}
