#pragma once 

#include "layeredOctree.h"
#include "layeredOctreeContainer.h"

#include <cuda_runtime.h>
#include <cudaHelpers.h>

template <typename T>
class LayeredOctreeContainerCuda {
    // Array of pointers to gpu memory
    LayeredOctree<T>* layers[OCTREE_MAX_DEPTH];
    // Array of pointers to gpu memory, stored on the gpu
    LayeredOctree<T>** layersGpu;
    int layerSizes[OCTREE_MAX_DEPTH];

public:
    LayeredOctreeContainerCuda(LayeredOctreeContainer<T>& container);
    ~LayeredOctreeContainerCuda();

    __device__ LayeredOctree<T>* getNode(int layer, int idx);
};

/*
 * Copy-constructor that copies data to CUDA memory
 */
template <typename T>
LayeredOctreeContainerCuda<T>::LayeredOctreeContainerCuda(LayeredOctreeContainer<T>& container) {
    for(int i = 0; i < OCTREE_MAX_DEPTH; i++) {
        int layerSize = container.getLayerSize(i);
        this->layerSizes[i] = layerSize;
        if(layerSize == 0) {
            this->layers[i] = nullptr;
        } else {
            //std::cout << "LayerSize(" << i << "): " << layerSize << std::endl;
            int layerByteCount = layerSize * sizeof(LayeredOctree<T>);
            std::cout << "Allocating " << (layerByteCount / 1024) << "KB (Layersize " << layerSize << ", bytes " << layerByteCount << ")" << std::endl;
            CUDA_CATCH_ERROR
            cudaMalloc(&this->layers[i], layerByteCount);
            CUDA_CATCH_ERROR
            cudaMemcpy(this->layers[i], container.getNode(i, 0), layerByteCount, cudaMemcpyHostToDevice);
            CUDA_CATCH_ERROR
        }
    }
    cudaMalloc(&this->layersGpu, OCTREE_MAX_DEPTH * sizeof(LayeredOctree<T>*));
    CUDA_CATCH_ERROR
    cudaMemcpy(this->layersGpu, this->layers, sizeof(this->layers), cudaMemcpyHostToDevice);
    CUDA_CATCH_ERROR
}

// Device-only to prevent accidental use on host
template <typename T>
__device__ LayeredOctree<T>* LayeredOctreeContainerCuda<T>::getNode(int layer, int idx) {
    return this->layersGpu[layer][idx];
}

template <typename T>
LayeredOctreeContainerCuda<T>::~LayeredOctreeContainerCuda() {
    std::cout << "Deconstructing layered octree" << std::endl;
    CUDA_CATCH_ERROR
    for(int i = 0; i < OCTREE_MAX_DEPTH; i++) {
        if(this->layers[i] != nullptr) {
            printf("Free: %p\n", (const void*)this->layers[i]);
            cudaFree(this->layers[i]);
            CUDA_CATCH_ERROR
        }
    }
    cudaFree(this->layersGpu);
    CUDA_CATCH_ERROR
}