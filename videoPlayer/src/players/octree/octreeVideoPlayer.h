#pragma once

#include <filesystem>

#include "../videoPlayer.h"

#include "loader/octreeLoader.h"
#include "render/octreeRenderer.h"

#include "layeredOctreeAllocator.h"
#include "octreeMetadata.h"


class OctreeVideoPlayer : public VideoPlayer {
    std::filesystem::path videoPath;

    OctreeMetadata metadata;
    LayeredOctreeAllocator allocator;
    OctreeLoader loader;

    loadedOctreeType* currentFrame = nullptr;
    OctreeFrameset* currentFrameset = nullptr;
    OctreeRenderer* renderer = nullptr;

    void drawDebugInfo();

public:
    OctreeVideoPlayer(TimeProvider* timeProvider, std::filesystem::path videoPath);
    ~OctreeVideoPlayer();

    void render(int width, int height, glm::mat4 view, glm::mat4 projection);

    float getVideoLength();

    void getVideoMetadata(VideoMetadata* metadata) const;

    bool isBuffering();
};