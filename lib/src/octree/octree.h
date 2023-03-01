#pragma once

#include <cstdint>
#include <glm/vec3.hpp>

#define ADDRESS_OCTREE(x, y, z) (x & 1) | ((y & 1) << 1) | ((z & 1) << 2)

template <typename T, typename C>
class Octree {
protected:
    C children[8];
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
    C getChildByIdx(int idx) const {
        return this->children[idx];
    }

    int getHashKey();

    T* getPayload() {
        return &this->payload;
    }
    uint8_t getChildCount() const {
        return this->childCount;
    }
    uint8_t getChildFlags() const {
        return this->childFlags;
    }
    uint8_t getLeafFlags() const {
        return this->leafFlags;
    }
};

template <typename T, typename C>
Octree<T, C>::Octree(T payload) {
    this->payload = payload;
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
    for(int i = 0; i < 8; i++) {
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
