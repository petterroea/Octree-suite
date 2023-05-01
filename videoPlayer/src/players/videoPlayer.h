#pragma once

#include <glm/mat4x4.hpp>

struct VideoMetadata {
    float length;
    int frameCount;
    float fps;
};

class VideoPlayer {

public:
    virtual void render(int width, int height, glm::mat4 view, glm::mat4 projection) = 0;

    virtual float getVideoLength() = 0;
    virtual bool isBuffering() = 0;

};