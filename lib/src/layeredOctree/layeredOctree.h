#pragma once

#include <stdint.h>
#include "../octree/octree.h"
#include "../octree/pointerOctree.h"

//#include "layeredOctreeContainer.h"

#define NO_NODE -1
typedef int layer_ptr_type;

template <typename T>
class LayeredOctree : public Octree<T, layer_ptr_type> {
public:
    LayeredOctree(T payload);

    bool setChild(layer_ptr_type child, int idx, bool isLeaf) {
        bool removed_existing = false;
        if(this->children[idx] != NO_NODE) {
            removed_existing = true;
        } else {
            this->childCount++;
        }

        this->children[idx] = child;
        this->childFlags |= 1 << idx;
        if(!isLeaf) {
            this->leafFlags |= 1 << idx;
        }
        return removed_existing;
    }
};

template <typename T>
LayeredOctree<T>::LayeredOctree(T payload) : Octree<T, layer_ptr_type>(payload) {
    for(int i = 0; i < 8; i++) {
        this->children[i] = NO_NODE;
    }
}

template <typename T>
void getOctreeStats(const PointerOctree<T>* tree, int* statsArray, int maxdepth, int curdepth) {
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
