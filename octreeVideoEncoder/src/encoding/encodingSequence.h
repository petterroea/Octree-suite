#pragma once

#include <octree/octree.h>

#include <vector>
#include <filesystem>
#include <fstream>

#include "../octreeSequence.h"
#include "octreeHashmap.h"
#include "deduplicator.h"
#include "../structures/layeredOctreeProcessingContainer.h"
#include "../videoEncoderRunArgs.h"


struct OctreeNode {

};

class EncodingSequence {
    OctreeSequence* sequence;
    DeDuplicator* deduplicator;
    VideoEncoderRunArgs* args;

    OctreeHashmap hashmaps[OCTREE_MAX_DEPTH];
    std::string fullPath;

    int from;
    int to;

    void populateHashmap(int depth, int roodIdx, LayeredOctreeProcessingContainer<octreeColorType>& octreeContainer, int max_depth);
    void writeToDisk(LayeredOctreeProcessingContainer<octreeColorType>& trees, std::string filename);
    void writeLayerToDisk(LayeredOctreeProcessingContainer<octreeColorType>& tree, std::ofstream& file, int i, int nodeCount, int childPtrCount);
    int write_compressed(unsigned char* data, int length, std::ofstream& file);
public:
    EncodingSequence(OctreeSequence* sequence, int from, int to, std::string fullPath, VideoEncoderRunArgs* args);
    ~EncodingSequence();
    void encode();
};