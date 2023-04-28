#pragma once

#include <cstdint>
#include <glm/vec3.hpp>

#include <cuda_runtime.h>

#define OCTREE_SIZE 8
#define ADDRESS_OCTREE(x, y, z) (x & 1) | ((y & 1) << 1) | ((z & 1) << 2)

template <typename T, typename C>
class Octree {
protected:
    C children[OCTREE_SIZE];
    T payload;
    uint8_t childCount = 0;
    // Is there a cild at this position?
    uint8_t childFlags = 0;
    // Is the child a leaf?
    uint8_t leafFlags = 0;
public:
    Octree(T payload);

    C getChildByCoords(int x, int y, int z) const {
        return this->getChildByIdx(ADDRESS_OCTREE(x, y, z));
    }
    __host__ __device__ C getChildByIdx(int idx) const {
        return this->children[idx];
    }

    int getHashKey();

    __host__ __device__ T* getPayload() {
        return &this->payload;
    }
    __host__ __device__ uint8_t getChildCount() const {
        return this->childCount;
    }
    __host__ __device__ uint8_t getChildFlags() const {
        return this->childFlags;
    }
    __host__ __device__ uint8_t getLeafFlags() const {
        return this->leafFlags;
    }

    __host__ void setChildFlags(uint8_t flags) {
        this->childFlags = flags;
    }
    __host__ void setLeafFlags(uint8_t flags) {
        this->leafFlags = flags;
    }
};

template <typename T, typename C>
Octree<T, C>::Octree(T payload): payload(payload) {
}

/*
template <typename T, typename C>
Octree<T, C>::~Octree() {
    for(int i = 0; i < 8; i++) {
        if(children[i] != nullptr) {
            delete children[i];
        }
    }
}
*/

template <typename T, typename C>
void getOctreeStats(const C tree, int* statsArray, int maxdepth, int curdepth) {
    if(curdepth == maxdepth) {
        return;
    }
    statsArray[curdepth]++;
    for(int i = 0; i < OCTREE_SIZE; i++) {
        auto child = tree->getChildByIdx(i);
        if(child) {
            getOctreeStats(child, statsArray, maxdepth, curdepth+1);
        }
    }
}

template <typename T, typename C>
int Octree<T, C>::getHashKey() {
    return this->childFlags;
}
