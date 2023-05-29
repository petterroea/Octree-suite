/*
 * Layered octree
 *
 * Octrees, but pointers are offsets into layer arrays
 * More than one octree can be stored in one structure
*/
#pragma once

#include <iostream>
#include <vector>
#include <exception>

#include "../octree/pointerOctree.h"
#include "layeredOctree.h"

#include "config.h"

template <typename T>
class LayeredOctreeContainerVector {
protected:
    std::vector<LayeredOctree<T>>* layers;
public:
    LayeredOctreeContainerVector();
    ~LayeredOctreeContainerVector();

    LayeredOctree<T>* getNode(int layer, int idx);
    int getLayerSize(int layer);
};

template <typename T>
LayeredOctreeContainerVector<T>::LayeredOctreeContainerVector() {
    this->layers = new std::vector<LayeredOctree<T>>[OCTREE_MAX_DEPTH];
    for(int i = 0; i < OCTREE_MAX_DEPTH; i++) {
        this->layers[i] = std::vector<LayeredOctree<T>>();
    }
}

template <typename T>
LayeredOctree<T>* LayeredOctreeContainerVector<T>::getNode(int layer, int idx) {
    if(idx > this->layers[layer].size()) {
        throw std::runtime_error("IDX too big: " + std::to_string(idx) + " > " + std::to_string(this->layers[layer].size()));
    }
    return &this->layers[layer][idx];
}

template <typename T>
int LayeredOctreeContainerVector<T>::getLayerSize(int layer) {
    return this->layers[layer].size();
}

template <typename T>
LayeredOctreeContainerVector<T>::~LayeredOctreeContainerVector() {

}

