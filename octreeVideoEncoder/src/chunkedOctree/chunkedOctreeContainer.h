/*
 * Chunked Octree
 *
 * Octrees, but pointers are offsets into layer arrays
 * More than one octree can be stored in one structure
*/
#pragma once

#include <octree/octree.h>
#include "layeredOctreeNode.h"

#define OCTREE_MAX_DEPTH OCTREE_MAX_DEPTH

template <typename T>
class ChunkedOctreeContainer {
    int chunkEndLevel;
    std::vector<ChunkedOctree<T>>* chunks;

    int installNode(Octree<T>* node, int level, int maxLevel);
public:
    ChunkedOctreeContainer(Octree<T>* originalOctree);
    ~ChunkedOctreeContainer();

    int addOctree(Octree<T>* node);

    ChunkedOctree<T>* getNode(int layer, int idx);

    int getChunkEndLevel();
};

template <typename T>
ChunkedOctreeContainer<T>::ChunkedOctreeContainer(Octree<T>* originalOctree) {
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

    this->chunks = new std::vector<ChunkedOctree<T>>[depth];

    // Calculate information used to decide where to make chunks
    for(int i = 0; i < OCTREE_MAX_DEPTH; i++) {
        std::cout << "Level " << i << ": " << stats[i] << std::endl;
    }
    this->installNode(originalOctree, 0, depth);
    std::cout << "Successfully converted octree to layered octree" << std::endl;
}

template <typename T>
int ChunkedOctreeContainer<T>::addOctree(Octree<T>* octree) {
    return this->installNode(octree, 0, OCTREE_MAX_DEPTH);
}

// Returns the array index the node was installed at
template <typename T>
int ChunkedOctreeContainer<T>::installNode(Octree<T>* node, int level, int maxLevel) {
    if(level == maxLevel) {
        std::cout << "WARNING: Unable to install entire octree due to the tree being too deep" << std::endl;
        throw "";
    }
    auto chunked = ChunkedOctree<T>(*node->getPayload(), level);
    for(int i = 0; i < 8; i++) {
        auto child = node->getChildByIdx(i);
        if(child) {
            int childIndex = this->installNode(child, level+1, maxLevel);
            chunked.setChild(childIndex, i);
        }
    }
    this->chunks[level].push_back(chunked);
    return this->chunks[level].size()-1;
}

template <typename T>
ChunkedOctree<T>* ChunkedOctreeContainer<T>::getNode(int layer, int idx) {
    return &this->chunks[layer][idx];
}

template <typename T>
ChunkedOctreeContainer<T>::~ChunkedOctreeContainer() {

}