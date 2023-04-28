#pragma once

#include <filesystem>
#include <chrono>

#include "../videoPlayer.h"

#include "loader/octreeLoader.h"
#include "render/octreeRenderer.h"

#include "layeredOctreeAllocator.h"
#include "octreeMetadata.h"


class OctreeVideoPlayer : public VideoPlayer {
    std::filesystem::path videoPath;

    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    float startOffset;
    bool playing;

    OctreeMetadata metadata;
    LayeredOctreeAllocator allocator;
    OctreeLoader loader;

    loadedOctreeType* currentFrame = nullptr;
    OctreeFrameset* currentFrameset = nullptr;
    OctreeRenderer* renderer = nullptr;

public:
    OctreeVideoPlayer(std::filesystem::path videoPath);
    ~OctreeVideoPlayer();

    void render(int width, int height, glm::mat4 view, glm::mat4 projection);
    void play();
    void pause();
    void seek(float time);

    float getTime();
    int getCurrentFrame();
    bool isPlaying() { return this->playing; }

    float getVideoLength();
    float getFps();
};