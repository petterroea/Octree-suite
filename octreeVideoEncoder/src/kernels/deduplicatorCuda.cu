#include "deduplicatorCuda.h"
#include <layeredOctree/treeComparison.h>

#include <iostream>

#define THREADS_PER_BLOCK 32
#define NODES_PER_THREAD 64

__global__ void kernel_nearnessLookupGenerator(float* nearnessTable, int jobsPerThread, layer_ptr_type* population, int populationSize, int layer, LayeredOctreeContainerCuda<octreeColorType>* container) {
    for(int i = 0; i < jobsPerThread; i++) {
        int idx = (blockIdx.x * THREADS_PER_BLOCK + threadIdx.x) * jobsPerThread + i;
        if(idx > populationSize * populationSize) {
            return;
        }
        int x = idx % populationSize;
        int y = idx / populationSize;
        if(y > populationSize) {
            y = 0 / 0; // break shit
        }

        float nearness = layeredOctreeSimilarity<LayeredOctreeContainerCuda<octreeColorType>>(population[x], population[y], layer, container);
        //float nearness = 0.0f;
        nearnessTable[idx] = nearness;
    }
}

void buildSimilarityLookupTableCuda(float* nearnessTable, std::vector<layer_ptr_type>& population, int layer, LayeredOctreeContainerCuda<octreeColorType>* container) {
    int populationSize = population.size();
    int jobs = populationSize*populationSize;

    // TODO scale this somehow
    int warpCount = 112; // RTX 3060 has 3584 cores
    int threadCount = warpCount * THREADS_PER_BLOCK;
    int jobsPerThread = (jobs + 1) / threadCount;

    int nearnessTableByteCount = sizeof(float) * populationSize * populationSize;

    //Please cuda can we have some more stack?
    cudaThreadSetLimit(cudaLimitStackSize, 1024*10);

    // Generate nearness table output
    float* nearnessTableCuda; 
    cudaMalloc(&nearnessTableCuda, nearnessTableByteCount);
    CUDA_CATCH_ERROR

    // Allocate population array on GPU and transfer
    layer_ptr_type* populationCuda;
    cudaMalloc(&populationCuda, sizeof(layer_ptr_type) * populationSize);
    CUDA_CATCH_ERROR
    cudaMemcpy(populationCuda, &population[0], sizeof(layer_ptr_type) * populationSize, cudaMemcpyHostToDevice);
    CUDA_CATCH_ERROR

    //Copy the CUDA container to CUDA memory
    LayeredOctreeContainerCuda<octreeColorType>* containerCuda;
    cudaMalloc(&containerCuda, sizeof(LayeredOctreeContainerCuda<octreeColorType>));
    CUDA_CATCH_ERROR
    cudaMemcpy(containerCuda, container, sizeof(LayeredOctreeContainerCuda<octreeColorType>), cudaMemcpyHostToDevice);
    CUDA_CATCH_ERROR

    // Start CUDA kernel
    std::cout << "Firing up CUDA to solve " << jobs << " jobs (" << jobsPerThread << " jobs per thread)" << std::endl;
    CUDA_CATCH_ERROR
    cudaDeviceSynchronize();
    kernel_nearnessLookupGenerator<<<warpCount, THREADS_PER_BLOCK>>>(nearnessTableCuda, jobsPerThread, populationCuda, populationSize, layer, containerCuda);
    cudaDeviceSynchronize();
    CUDA_CATCH_ERROR
    std::cout << jobs << ": Done, copying back" << std::endl;

    // Copy nearness table output back
    cudaMemcpy(nearnessTable, nearnessTableCuda, nearnessTableByteCount, cudaMemcpyDeviceToHost);
    CUDA_CATCH_ERROR
    std::cout << jobs << ": copy done" << std::endl;

    // Free memory
    cudaFree(nearnessTableCuda);
    cudaFree(populationCuda);
    cudaFree(containerCuda);
    CUDA_CATCH_ERROR
}
