#pragma once

#include <cstdint>
#include <glm/vec3.hpp>

#define ADDRESS_OCTREE(x, y, z) (x & 1) | ((y & 1) << 1) | ((z & 1) << 2)

template <typename T>
class Octree {
    Octree<T>* children[8] {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    T payload;
    uint8_t childCount = 0;
    // Is there a cild at this position?
    uint8_t childFlags = 0;
    // Is the child a leaf?
    uint8_t leafFlags = 0;
public:
    Octree(T payload);
    ~Octree();

    Octree<T>* getChildByCoords(int x, int y, int z) const {
        return this->getChildByIdx(ADDRESS_OCTREE(x, y, z));
    }
    Octree<T>* getChildByIdx(int idx) const {
        return this->children[idx];
    }

    int getHashKey();

    T* getPayload() {
        return &this->payload;
    }
    bool setChild(Octree<T>* child, int idx) {
        bool removed_existing = false;
        if(this->children[idx] != nullptr) {
            delete this->children[idx];
            removed_existing = true;
        } else {
            this->childCount++;
        }
        this->children[idx] = child;
        this->childFlags |= 1 << idx;
        if(!child->childCount) {
            this->leafFlags |= 1 << idx;
        }
        return removed_existing;
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

template <typename T>
Octree<T>::Octree(T payload) {
    this->payload = payload;
}

template <typename T>
Octree<T>::~Octree() {
    for(int i = 0; i < 8; i++) {
        if(children[i] != nullptr) {
            delete children[i];
        }
    }
}

float diffOctreeColor(Octree<glm::vec3>* lhs, Octree<glm::vec3>* rhs);
float octreeSimilarity(const Octree<glm::vec3>* lhs, const Octree<glm::vec3>* rhs);
float octreeFillRate(const Octree<glm::vec3>* tree);

template <typename T>
void getOctreeStats(const Octree<T>* tree, int* statsArray, int maxdepth, int curdepth) {
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

template <typename T>
int Octree<T>::getHashKey() {
    return this->childFlags;
    int key = 0;
    for(int i = 0; i < 8; i++) {
        key |= (this->getChildByIdx(i) != nullptr ? 1 : 0) << i;
    }
    return key;
}
