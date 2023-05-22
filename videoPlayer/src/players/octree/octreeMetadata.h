#pragma once

#include <filesystem>
#include <vector>

#include "octreeFrameset.h"


class OctreeMetadata {
    float fps;
    int frameCount;

    std::filesystem::path path;

    std::vector<OctreeFrameset*> frames;
public:
    OctreeMetadata(std::filesystem::path path);
    ~OctreeMetadata();
    
    float getFps() const;
    int getFrameCount() const;
    std::filesystem::path getPath() const;
    OctreeFrameset* getFramesetByFrame(int index) const;
};