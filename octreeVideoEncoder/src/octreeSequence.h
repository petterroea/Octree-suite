#pragma once

#include <glm/vec3.hpp>

#include <octree/pointerOctree.h>

#include <filesystem>

#include "config.h"

class OctreeSequence {
    std::filesystem::path sequenceFolder;
    int frameCount;
    float fps;
public:
    OctreeSequence(std::filesystem::path videoFolder);
    ~OctreeSequence();

    PointerOctree<octreeColorType>* getOctree(int frame);

    int getFrameCount () { return this->frameCount; }
    float getFps() { return this->fps; }
};