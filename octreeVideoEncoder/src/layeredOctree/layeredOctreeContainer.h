/*
 * Chunked Octree
 *
 * Octrees, but pointers are offsets into layer arrays
 * More than one octree can be stored in one structure
*/
#pragma once

#include <iostream>
#include <vector>

#include <octree/pointerOctree.h>
#include "layeredOctree.h"

#define OCTREE_MAX_DEPTH 20

template <typename T>
class LayeredOctreeContainer {
    int chunkEndLevel;
    std::vector<LayeredOctree<T>>* chunks;

    int installNode(PointerOctree<T>* node, int level, int maxLevel);
public:
    LayeredOctreeContainer(PointerOctree<T>* originalOctree);
    ~LayeredOctreeContainer();

    int addOctree(PointerOctree<T>* node);

    LayeredOctree<T>* getNode(int layer, int idx);

    int getChunkEndLevel();
};

template <typename T>
LayeredOctreeContainer<T>::LayeredOctreeContainer(PointerOctree<T>* originalOctree) {
    int stats[OCTREE_MAX_DEPTH] {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    getOctreeStats(originalOctree, stats, OCTREE_MAX_DEPTH, 0);
    std::cout << "Octree stats:" << std::endl;

    int depth = 0;
    for(int i = 0; i < OCTREE_MAX_DEPTH; i++) {
        if(stats[i] == 0) {
            depth = i;
            break;
        }
    }
    std::cout << "Tree depth: " << depth << std::endl;
    depth = depth + 2; // Some margin of error :)

    this->chunks = new std::vector<LayeredOctree<T>>[depth];

    // Calculate information used to decide where to make chunks
    for(int i = 0; i < OCTREE_MAX_DEPTH; i++) {
        std::cout << "Level " << i << ": " << stats[i] << std::endl;
    }
    this->installNode(originalOctree, 0, depth);
    std::cout << "Successfully converted octree to layered octree" << std::endl;
}

template <typename T>
int LayeredOctreeContainer<T>::addOctree(PointerOctree<T>* octree) {
    return this->installNode(octree, 0, OCTREE_MAX_DEPTH);
}

// Returns the array index the node was installed at
template <typename T>
int LayeredOctreeContainer<T>::installNode(PointerOctree<T>* node, int level, int maxLevel) {
    if(level == maxLevel) {
        std::cout << "WARNING: Unable to install entire octree due to the tree being too deep" << std::endl;
        throw "";
    }
    auto chunked = LayeredOctree<T>(*node->getPayload(), level);
    for(int i = 0; i < 8; i++) {
        auto child = node->getChildByIdx(i);
        if(child) {
            int childIndex = this->installNode(child, level+1, maxLevel);
            chunked.setChild(childIndex, i, child->getChildCount() == 0);
        }
    }
    this->chunks[level].push_back(chunked);
    return this->chunks[level].size()-1;
}

template <typename T>
LayeredOctree<T>* LayeredOctreeContainer<T>::getNode(int layer, int idx) {
    return &this->chunks[layer][idx];
}

template <typename T>
LayeredOctreeContainer<T>::~LayeredOctreeContainer() {

}

// Helper functions

float diffLayeredOctreeColor(layer_ptr_type lhs, layer_ptr_type rhs, int layer, LayeredOctreeContainer<glm::vec3>& container);
float layeredOctreeSimilarity(layer_ptr_type lhs, layer_ptr_type rhs, int layer, LayeredOctreeContainer<glm::vec3>& container);
float layeredOctreeFillRate(layer_ptr_type tree, int layer, LayeredOctreeContainer<glm::vec3>& container);
