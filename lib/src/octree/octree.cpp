#include "octree.h"

#include <bit>

#include <glm/geometric.hpp>

float diffOctreeColor(Octree<glm::vec3>* lhs, Octree<glm::vec3>* rhs) {
    // Use euclidean distance. TODO: Delta E
    return glm::length(*rhs->getPayload() - *lhs->getPayload());
}

// **Structural** octree similarity
float octreeSimilarity(const Octree<glm::vec3>* lhs, const Octree<glm::vec3>* rhs) {
    int lhs_children = lhs->getChildCount();
    int rhs_children = rhs->getChildCount();

    if(!lhs_children && !rhs_children) {
        // Both leaf nodes? 100% similar
        return 1.0f;
    } else if(lhs_children && !rhs_children) {
        return octreeFillRate(lhs);
    } else if(!lhs_children && rhs_children) {
        return octreeFillRate(rhs);
    }
    // Both nodes are populated - calculate similarity
    float sum = 0.0f;
    for(int i = 0; i < 8; i++) {
        auto lhs_child = lhs->getChildByIdx(i);
        auto rhs_child = rhs->getChildByIdx(i);
        if(lhs_child == nullptr && rhs_child == nullptr) {
            sum += 1.0f;
        } else if(lhs_child != nullptr && rhs_child != nullptr) {
            sum += octreeSimilarity(lhs_child, rhs_child);
        } else if(lhs_child != nullptr && rhs_child == nullptr) {
            sum += 1.0f - octreeFillRate(lhs_child);
        } else { //lhs_child == nullptr && rhs_child != nullptr
            sum += 1.0f - octreeFillRate(rhs_child);
        }
    }
    return sum / 8.0f;
}

float octreeFillRate(const Octree<glm::vec3>* tree) {
    if(!tree->getChildCount()) {
        return 1.0f;
    }
    /*if(
        lhs->getLeafFlags() == rhs->getLeafFlags() && 
        lhs->getChildFlags() == rhs->getChildFlags() &&
        lhs->getLeafFlags() ^ lhs->getChildFlags() == 0) {
        return 1.0f;
    } */
    auto childFlags = tree->getChildFlags();
    auto leafFlags = tree->getLeafFlags();
    //All leafs? Use bit magic instead
    if( leafFlags == childFlags ) {
        return (static_cast<float>(std::popcount(childFlags)) / 8.0f);
    }

    float sum = 0.0f;
    sum += std::popcount(leafFlags);

    #pragma GCC unroll 8
    for(int i = 0; i < 8; i++) {
        if(((childFlags ^ leafFlags) >> i) & 1) {
            sum += octreeFillRate(tree->getChildByIdx(i));
        }
    }
    return sum / 8.0f;
}