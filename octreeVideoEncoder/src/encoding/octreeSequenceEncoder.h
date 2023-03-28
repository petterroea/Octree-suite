#pragma once

#include <filesystem>

#include "../octreeSequence.h"

class OctreeSequenceEncoder {
    OctreeSequence* sequence;
    std::filesystem::path outputFolder;

public:
    OctreeSequenceEncoder(OctreeSequence* sequence, std::filesystem::path outputFolder);
    ~OctreeSequenceEncoder();

    void encode();
};