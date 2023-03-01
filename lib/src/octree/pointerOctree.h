#pragma once

#include <cstring>

#include "octree.h"

template <typename T>
class PointerOctree : public Octree<T, PointerOctree<T>*> {
public:
    PointerOctree(T payload);
    ~PointerOctree();

    bool setChild(PointerOctree<T>* child, int idx) {
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
};

float diffPointerOctreeColor(PointerOctree<glm::vec3>* lhs, PointerOctree<glm::vec3>* rhs);
float pointerOctreeSimilarity(PointerOctree<glm::vec3>* lhs, PointerOctree<glm::vec3>* rhs);
float pointerOctreeFillRate(PointerOctree<glm::vec3>* tree);

template <typename T>
PointerOctree<T>::PointerOctree(T payload) : Octree<T, PointerOctree<T>*>(payload) {
    memset(this->children, 0, sizeof(this->children));

}

template <typename T>
PointerOctree<T>::~PointerOctree() {
    for(int i = 0; i < 8; i++) {
        if(this->children[i] != nullptr) {
            delete this->children[i];
        }
    }
}