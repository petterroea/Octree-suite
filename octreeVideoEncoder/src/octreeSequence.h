#pragma once

#include <glm/vec3.hpp>

#include <octree/pointerOctree.h>

#include <filesystem>

class OctreeSequence {
    std::filesystem::path sequenceFolder;
    int frameCount;
    float fps;
public:
    OctreeSequence(std::filesystem::path videoFolder);
    ~OctreeSequence();

    PointerOctree<glm::vec3>* getOctree(int frame);

    int getFrameCount () { return this->frameCount; }
    float getFps() { return this->fps; }
};