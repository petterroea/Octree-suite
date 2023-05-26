#pragma once

#include <layeredOctree/layeredOctreeContainerVector.h>

#include "octreeProcessingPayload.h"

template<typename T>
class LayeredOctreeProcessingContainer: public LayeredOctreeContainerVector<OctreeProcessingPayload<T>> {
    int installNode(PointerOctree<T>* node, int level, int maxLevel);
public:
    int addOctree(PointerOctree<T>* node);
};

template <typename T>
int LayeredOctreeProcessingContainer<T>::addOctree(PointerOctree<T>* octree) {
    return this->installNode(octree, 0, OCTREE_MAX_DEPTH);
}

// Returns the array index the node was installed at
template <typename T>
int LayeredOctreeProcessingContainer<T>::installNode(PointerOctree<T>* node, int level, int maxLevel) {
    if(level == maxLevel) {
        std::cout << "WARNING: Unable to install entire octree due to the tree being too deep" << std::endl;
        throw "";
    }
    OctreeProcessingPayload<octreeColorType> payload(*node->getPayload());
    auto chunked = LayeredOctree<OctreeProcessingPayload<T>>(payload);
    for(int i = 0; i < 8; i++) {
        auto child = node->getChildByIdx(i);
        if(child) {
            int childIndex = this->installNode(child, level+1, maxLevel);
            chunked.setChild(childIndex, i, child->getChildCount() == 0);
        }
    }
    this->layers[level].push_back(chunked);
    return this->layers[level].size()-1;
}
