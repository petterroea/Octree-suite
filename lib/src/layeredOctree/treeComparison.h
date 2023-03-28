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
__host__ __device__ float diffLayeredOctreeColor(layer_ptr_type lhs, layer_ptr_type rhs, int layer, T container) {
    // Use euclidean distance. TODO: Delta E
    return glm::length(*container.getNode(layer, lhs)->getPayload() - *container.getNode(layer, rhs)->getPayload());
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
    //All leafs? Use bit magic instead
    if( leafFlags == childFlags ) {
        return (static_cast<float>(popcount(childFlags)) / 8.0f);
    }

    float sum = 0.0f;
    sum += popcount(leafFlags);

    #pragma GCC unroll 8
    for(int i = 0; i < 8; i++) {
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
        return layeredOctreeFillRate(lhs, layer, container);
    } else if(!lhs_children && rhs_children) {
        return layeredOctreeFillRate(rhs, layer, container);
    }
    // Both nodes are populated - calculate similarity
    float sum = 0.0f;
    for(int i = 0; i < 8; i++) {
        auto lhs_child = lhs_node->getChildByIdx(i);
        auto rhs_child = rhs_node->getChildByIdx(i);
        if(lhs_child == NO_NODE && rhs_child == NO_NODE) {
            sum += 1.0f;
        } else if(lhs_child != NO_NODE && rhs_child != NO_NODE) {
            sum += layeredOctreeSimilarity<T>(lhs_child, rhs_child, layer+1, container);
        } else if(lhs_child != NO_NODE && rhs_child == NO_NODE) {
            sum += 1.0f - layeredOctreeFillRate<T>(lhs_child, layer+1, container);
        } else { //lhs_child == NO_NODE && rhs_child != NO_NODE
            sum += 1.0f - layeredOctreeFillRate<T>(rhs_child, layer+1, container);
        }
    }
    return sum / 8.0f;
}
