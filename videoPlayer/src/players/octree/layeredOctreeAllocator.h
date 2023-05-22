#pragma once

#include <vector>

#include <layeredOctree/layeredOctreeContainerStatic.h>

#include "config.h"

// Allows us to more easily handle allocation of layered octrees in the future.
class LayeredOctreeAllocator {
public:
    LayeredOctreeAllocator();

    LayeredOctreeContainerStatic<octreeColorType>* allocate(std::vector<int>& layerSizes);
    void free(LayeredOctreeContainerStatic<octreeColorType>* container);
};