#pragma once
#include "layeredOctree.h"

#ifdef __CUDACC__
#define popcount __popc
#else

#include <bit>
#define popcount std::popcount

#endif

#include <glm/geometric.hpp>

template <typename T>
__host__ __device__ float diffLayeredOctreeColor(layer_ptr_type lhs, layer_ptr_type rhs, int layer, T* container) {
    // Use euclidean distance. TODO: Delta E
    return glm::length(*container->getNode(layer, lhs)->getPayload() - *container->getNode(layer, rhs)->getPayload());
}
template <typename T>
__host__ __device__ float diffProcessingLayeredOctreeColor(layer_ptr_type lhs, layer_ptr_type rhs, int layer, LayeredOctreeProcessingContainer<T>* container) {
    // Use euclidean distance. TODO: Delta E
    return glm::length(container->getNode(layer, lhs)->getPayload()->data - container->getNode(layer, rhs)->getPayload()->data);
}

template <typename T>
__host__ __device__ float layeredOctreeFillRate(layer_ptr_type tree, int layer, T* container) {
    if(layer == OCTREE_MAX_DEPTH) {
        #ifdef __CUDA_ARCH__
            return 0.0f;
        #else
            throw "Octree is too deep";
        #endif
    }
    auto tree_node = container->getNode(layer, tree);
    // No children? Leaf node - 100% fill
    if(!tree_node->getChildCount()) {
        return 1.0f;
    }
    /*if(
        lhs->getLeafFlags() == rhs->getLeafFlags() && 
        lhs->getChildFlags() == rhs->getChildFlags() &&
        lhs->getLeafFlags() ^ lhs->getChildFlags() == 0) {
        return 1.0f;
    } */
    auto childFlags = tree_node->getChildFlags();
    auto leafFlags = tree_node->getLeafFlags();

    //All children are leafs? Use bit magic instead
    if( leafFlags == childFlags ) {
        // How many % of the children are filled?
        return (static_cast<float>(popcount(childFlags)) / 8.0f);
    }

    // Not all children are leafs, we need to recurse
    float sum = 0.0f;
    // Add all leafs to the sum
    sum += static_cast<float>(popcount(leafFlags));

    #pragma GCC unroll 8
    for(int i = 0; i < OCTREE_SIZE; i++) {
        // Recurse all children that aren't leafs
        if(((childFlags ^ leafFlags) >> i) & 1) {
            sum += layeredOctreeFillRate<T>(tree_node->getChildByIdx(i), layer+1, container);
        }
    }
    return sum / 8.0f;
}

// **Structural** octree similarity
template <typename T>
__host__ __device__ float layeredOctreeSimilarity(layer_ptr_type lhs, layer_ptr_type rhs, int layer, T* container) {
    if(layer == OCTREE_MAX_DEPTH) {
        #ifdef __CUDA_ARCH__
            return 0.0f;
        #else
            throw "Octree is too deep";
        #endif
    }
    auto lhs_node = container->getNode(layer, lhs);
    auto rhs_node = container->getNode(layer, rhs);

    int lhs_children = lhs_node->getChildCount();
    int rhs_children = rhs_node->getChildCount();

    if(!lhs_children && !rhs_children) {
        // Both leaf nodes? 100% similar
        return 1.0f;
    } else if(lhs_children && !rhs_children) {
        // rhs has no children, meaning it's a leaf node.
        // The more space lhs occupies, the more similar the nodes are
        // TODO
        return layeredOctreeFillRate(lhs, layer, container);
    } else if(!lhs_children && rhs_children) {
        // Same as above but opposite
        return layeredOctreeFillRate(rhs, layer, container);
    }
    // Both nodes are populated - calculate similarity of the children
    float sum = 0.0f;
    for(int i = 0; i < 8; i++) {
        auto lhs_child = lhs_node->getChildByIdx(i);
        auto rhs_child = rhs_node->getChildByIdx(i);
        if(lhs_child == NO_NODE && rhs_child == NO_NODE) {
            // Both child spots are empty, 100% similar
            sum += 1.0f;
        } else if(lhs_child != NO_NODE && rhs_child != NO_NODE) {
            // Both child spots are occupied, get their similarity
            sum += layeredOctreeSimilarity<T>(lhs_child, rhs_child, layer+1, container);
        } else if(lhs_child != NO_NODE && rhs_child == NO_NODE) {
            // lhs child is occupied, rhs child is empty. They are 100% similar if lhs is empty
            sum += 1.0f - layeredOctreeFillRate<T>(lhs_child, layer+1, container);
        } else { //lhs_child == NO_NODE && rhs_child != NO_NODE
            // Only rhs child is occupied, fill rate is inverse of lhs fill rate
            sum += 1.0f - layeredOctreeFillRate<T>(rhs_child, layer+1, container);
        }
    }
    return sum / 8.0f;
}
