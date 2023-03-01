#include "pointerOctree.h"

#include <bit>

#include <glm/geometric.hpp>

float diffPointerOctreeColor(PointerOctree<glm::vec3>* lhs, PointerOctree<glm::vec3>* rhs) {
    // Use euclidean distance. TODO: Delta E
    return glm::length(*rhs->getPayload() - *lhs->getPayload());
}

// **Structural** octree similarity
float pointerOctreeSimilarity(PointerOctree<glm::vec3>* lhs, PointerOctree<glm::vec3>* rhs) {
    int lhs_children = lhs->getChildCount();
    int rhs_children = rhs->getChildCount();

    if(!lhs_children && !rhs_children) {
        // Both leaf nodes? 100% similar
        return 1.0f;
    } else if(lhs_children && !rhs_children) {
        return pointerOctreeFillRate(lhs);
    } else if(!lhs_children && rhs_children) {
        return pointerOctreeFillRate(rhs);
    }
    // Both nodes are populated - calculate similarity
    float sum = 0.0f;
    for(int i = 0; i < 8; i++) {
        auto lhs_child = lhs->getChildByIdx(i);
        auto rhs_child = rhs->getChildByIdx(i);
        if(lhs_child == nullptr && rhs_child == nullptr) {
            sum += 1.0f;
        } else if(lhs_child != nullptr && rhs_child != nullptr) {
            sum += pointerOctreeSimilarity(lhs_child, rhs_child);
        } else if(lhs_child != nullptr && rhs_child == nullptr) {
            sum += 1.0f - pointerOctreeFillRate(lhs_child);
        } else { //lhs_child == nullptr && rhs_child != nullptr
            sum += 1.0f - pointerOctreeFillRate(rhs_child);
        }
    }
    return sum / 8.0f;
}

float pointerOctreeFillRate(PointerOctree<glm::vec3>* tree) {
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
            sum += pointerOctreeFillRate(tree->getChildByIdx(i));
        }
    }
    return sum / 8.0f;
}
