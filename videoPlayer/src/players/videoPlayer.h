#pragma once

#include <glm/mat4x4.hpp>

#include "../time/timeProvider.h"

struct VideoMetadata {
    int frameCount;
    float fps;
};

class VideoPlayer {

protected:
    TimeProvider* timeProvider;
public:
    VideoPlayer(TimeProvider* timeProvider);
    virtual void render(int width, int height, glm::mat4 view, glm::mat4 projection) = 0;

    virtual float getVideoLength() = 0;
    virtual bool isBuffering() = 0;

    virtual void getVideoMetadata(VideoMetadata* metadata) const = 0;

    int getCurrentFrame();

};