#pragma once

#include "timeProvider.h"

class RecordingTimeProvider : public TimeProvider {
    int frameCounter = 0;

    float fps;
    int frameCount;
public:
    RecordingTimeProvider(float fps, int frameCount);

    float getTime() const;
    int getCurrentFrame() const;

    void setFrameCount(int newFrameCount);

    void renderControls();

    void onFrameComplete();
};