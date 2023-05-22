#include "layeredOctreeAllocator.h"

#include <iostream>

LayeredOctreeAllocator::LayeredOctreeAllocator() {

}

LayeredOctreeContainerStatic<octreeColorType>* LayeredOctreeAllocator::allocate(std::vector<int>& layerSizes) {
    if(layerSizes.size() > OCTREE_MAX_DEPTH) {
        std::cout << "Requested layer size: " << layerSizes.size() << " max size: " << OCTREE_MAX_DEPTH << std::endl;
        throw std::runtime_error("LayeredOctreeAllocator::allocate: layerSizes.size() > OCTREE_MAX_DEPTH");
    }
    auto container = new LayeredOctreeContainerStatic<octreeColorType>(layerSizes);
    return container;
}
void LayeredOctreeAllocator::free(LayeredOctreeContainerStatic<octreeColorType>* container) {
    delete container;
}