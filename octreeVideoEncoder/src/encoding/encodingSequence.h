#pragma once

#include <glm/vec3.hpp>
#include <octree/octree.h>

#include <vector>

#include "../octreeSequence.h"
#include "octreeHashmap.h"
#include "deduplicator.h"
#include "../layeredOctree/layeredOctreeContainer.h"

typedef glm::vec3 octreeProcessingPayload;

struct OctreeNode {

};

class EncodingSequence {
    OctreeSequence* sequence;
    DeDuplicator* deduplicator;

    OctreeHashmap hashmaps[OCTREE_MAX_DEPTH];

    int from;
    int to;

    void populateHashmap(int depth, int roodIdx, LayeredOctreeContainer<octreeProcessingPayload>& octreeContainer, int max_depth);
public:
    EncodingSequence(OctreeSequence* sequence, int from, int to);
    void encode();
};