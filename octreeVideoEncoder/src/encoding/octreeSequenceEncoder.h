#pragma once

#include "../octreeSequence.h"

class OctreeSequenceEncoder {
    OctreeSequence* sequence;
public:
    OctreeSequenceEncoder(OctreeSequence* sequence);
    ~OctreeSequenceEncoder();

    void encode();
};