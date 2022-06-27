#include "octree.h"


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

template <typename T>
Octree<T>* Octree<T>::getChild(int x, int y, int z) {
    return this->children[ADDRESS_OCTREE(x, y, z)];
}
template <typename T>
T* Octree<T>::getPayload() {
    return &this->payload;
}

template <typename T>
bool Octree<T>::setChild(Octree<T>* child, int x, int y, int z) {
    bool removed_existing = false;
    if(this->children[ADDRESS_OCTREE(x, y, z)] != nullptr) {
        delete this->children[ADDRESS_OCTREE(x, y, z)];
        removed_existing = true;
    }
    this->children[ADDRESS_OCTREE(x, y, z)] = child;
    return removed_existing;
}