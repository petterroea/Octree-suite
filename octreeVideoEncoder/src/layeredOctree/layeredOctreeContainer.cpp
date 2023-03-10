#include "layeredOctreeContainer.h"


// We can't use popcount inside cuda because it's a c++ 20 feature
// Cmake and CUDA does support c++20, but not in any release that has reached ubuntu yet.
#ifdef __CUDA_ARCH__
#define popcount __popc
#else

#include <bit>
#define popcount std::popcount

#endif

#include <glm/geometric.hpp>

float diffLayeredOctreeColor(layer_ptr_type lhs, layer_ptr_type rhs, int layer, LayeredOctreeContainer<glm::vec3>& container) {
    // Use euclidean distance. TODO: Delta E
    return glm::length(*container.getNode(layer, lhs)->getPayload() - *container.getNode(layer, rhs)->getPayload());
}

// **Structural** octree similarity
float layeredOctreeSimilarity(layer_ptr_type lhs, layer_ptr_type rhs, int layer, LayeredOctreeContainer<glm::vec3>& container) {
    if(layer == OCTREE_MAX_DEPTH) {
        throw "Octree is too deep";
    }
    auto lhs_node = container.getNode(layer, lhs);
    auto rhs_node = container.getNode(layer, rhs);

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
            sum += layeredOctreeSimilarity(lhs_child, rhs_child, layer+1, container);
        } else if(lhs_child != NO_NODE && rhs_child == NO_NODE) {
            sum += 1.0f - layeredOctreeFillRate(lhs_child, layer+1, container);
        } else { //lhs_child == NO_NODE && rhs_child != NO_NODE
            sum += 1.0f - layeredOctreeFillRate(rhs_child, layer+1, container);
        }
    }
    return sum / 8.0f;
}

float layeredOctreeFillRate(layer_ptr_type tree, int layer, LayeredOctreeContainer<glm::vec3>& container) {
    if(layer == OCTREE_MAX_DEPTH) {
        throw "Octree is too deep";
    }
    auto tree_node = container.getNode(layer, tree);
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
            sum += layeredOctreeFillRate(tree_node->getChildByIdx(i), layer+1, container);
        }
    }
    return sum / 8.0f;
}