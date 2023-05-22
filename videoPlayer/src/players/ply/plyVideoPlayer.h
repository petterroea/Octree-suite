#pragma once

#include <filesystem>

#include "../videoPlayer.h"
#include "../../time/timeProvider.h"

#include "plyMetadata.h"
#include "plyLoader.h"
#include "plyRenderer.h"

class PlyVideoPlayer : public VideoPlayer {
    PlyMetadata metadata;
    
    PlyLoader* loader;
    PlyRenderer* renderer;

    Pointcloud* currentRenderingPointcloud = nullptr;
    int currentRenderingFrame = -1;
    int lastRequestedSeek = 0;

    void seekPlyFrame();
    void setNewFrame(Pointcloud* pointcloud, int frame);
public:
    PlyVideoPlayer(TimeProvider* timeProvider, const std::filesystem::path folder);
    ~PlyVideoPlayer();
    void render(int width, int height, glm::mat4 view, glm::mat4 projection);

    float getVideoLength();
    bool isBuffering();

    void getVideoMetadata(VideoMetadata* metadata) const;
};