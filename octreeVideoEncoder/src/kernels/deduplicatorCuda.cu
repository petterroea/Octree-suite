#include "deduplicatorCuda.h"

#include <iostream>

#define THREADS_PER_BLOCK 32
#define NODES_PER_THREAD 64

__global__ void kernel_nearnessLookupGenerator(float* nearnessTable, int populationSize, int layer, LayeredOctreeContainerCuda<octreeProcessingPayload>& container) {
    int idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
    if(idx > populationSize * populationSize) {
        return;
    }
    int x = idx % populationSize;
    int y = idx / populationSize;

    float nearness = layeredOctreeSimilarity(x, y, layer, container);
    //float nearness = 0.0f;
    nearnessTable[x + y*populationSize] = nearness;
}

void buildSimilarityLookupTableCuda(float* nearnessTable, int populationSize, int layer, LayeredOctreeContainerCuda<octreeProcessingPayload>& container) {
    int jobs = populationSize*populationSize / THREADS_PER_BLOCK;
    int warpCount = (jobs+1) / THREADS_PER_BLOCK;

    int nearnessTableByteCount = sizeof(float) * populationSize * populationSize;
    float* nearnessTableCuda; 
    std::cout << "Firing up CUDA to solve " << jobs << " jobs" << std::endl;
    cudaMalloc(&nearnessTableCuda, nearnessTableByteCount);
    CUDA_CATCH_ERROR
    kernel_nearnessLookupGenerator<<<warpCount, THREADS_PER_BLOCK>>>(nearnessTableCuda, populationSize, layer, container);
    CUDA_CATCH_ERROR
    std::cout << jobs << ": Done, copying back" << std::endl;
    cudaMemcpy(nearnessTable, nearnessTableCuda, nearnessTableByteCount, cudaMemcpyDeviceToHost);
    CUDA_CATCH_ERROR
    cudaFree(nearnessTableCuda);
    CUDA_CATCH_ERROR
}
