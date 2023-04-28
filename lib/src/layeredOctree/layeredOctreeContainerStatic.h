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

#include "config.h"

template <typename T>
class LayeredOctreeContainerStatic {
protected:
    std::vector<int> layerCounts;
    LayeredOctree<T>** layers;
public:
    LayeredOctreeContainerStatic(std::vector<int>& layerCounts);
    ~LayeredOctreeContainerStatic();

    LayeredOctree<T>* getNode(int layer, int idx);
    int getLayerSize(int layer);
};

template <typename T>
LayeredOctreeContainerStatic<T>::LayeredOctreeContainerStatic(std::vector<int>& layerCounts) : layerCounts(layerCounts){
    this->layers = new LayeredOctree<T>*[layerCounts.size()];
    for(int i = 0; i < layerCounts.size(); i++){
        // Probably faster than default constructing everything
        this->layers[i] = (LayeredOctree<T>*)calloc(layerCounts[i], sizeof(LayeredOctree<T>));
    }
}

template <typename T>
LayeredOctree<T>* LayeredOctreeContainerStatic<T>::getNode(int layer, int idx) {
    return &this->layers[layer][idx];
}

template <typename T>
int LayeredOctreeContainerStatic<T>::getLayerSize(int layer) {
    return this->layerCounts[layer];
}

template <typename T>
LayeredOctreeContainerStatic<T>::~LayeredOctreeContainerStatic() {
    for(int i = 0; i < this->layerCounts.size(); i++){
        delete[] this->layers[i];
    }
    delete[] this->layers;
}
