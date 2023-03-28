/*
 * Chunked Octree
 *
 * Octrees, but pointers are offsets into layer arrays
 * More than one octree can be stored in one structure
*/
#pragma once

#include <iostream>
#include <vector>

#include "../octree/pointerOctree.h"
#include "layeredOctree.h"

#define OCTREE_MAX_DEPTH 20

template <typename T>
class LayeredOctreeContainer {
protected:
    std::vector<LayeredOctree<T>>* layers;
public:
    LayeredOctreeContainer();
    ~LayeredOctreeContainer();

    LayeredOctree<T>* getNode(int layer, int idx);
    int getLayerSize(int layer);
};

template <typename T>
LayeredOctreeContainer<T>::LayeredOctreeContainer() {
    this->layers = new std::vector<LayeredOctree<T>>[OCTREE_MAX_DEPTH];
    for(int i = 0; i < OCTREE_MAX_DEPTH; i++) {
        this->layers[i] = std::vector<LayeredOctree<T>>();
    }
}

template <typename T>
LayeredOctree<T>* LayeredOctreeContainer<T>::getNode(int layer, int idx) {
    return &this->layers[layer][idx];
}

template <typename T>
int LayeredOctreeContainer<T>::getLayerSize(int layer) {
    return this->layers[layer].size();
}

template <typename T>
LayeredOctreeContainer<T>::~LayeredOctreeContainer() {

}
