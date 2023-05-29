#pragma once

#include <octree/octree.h>

#include <vector>
#include <filesystem>

#include "../octreeSequence.h"
//#include "octreeHashmap.h"
#include "dctContainer.h"
#include "../structures/layeredOctreeProcessingContainer.h"
#include "../videoEncoderRunArgs.h"


struct OctreeNode {

};

class EncodingSequence {
    OctreeSequence* sequence;
    VideoEncoderRunArgs* args;
    DctContainer dctContainer;

    //OctreeHashmap hashmaps[OCTREE_MAX_DEPTH];
    std::string fullPath;

    int from;
    int to;

//    void populateHashmap(int depth, int roodIdx, LayeredOctreeProcessingContainer<octreeColorType>& octreeContainer, int max_depth);
    void writeToDisk(LayeredOctreeProcessingContainer<octreeColorType>& trees, std::string filename);
public:
    EncodingSequence(OctreeSequence* sequence, int from, int to, std::string fullPath, VideoEncoderRunArgs* args);
    ~EncodingSequence();
    void encode();
};