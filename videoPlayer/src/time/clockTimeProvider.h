#pragma once

#include <chrono>

#include "timeProvider.h"

class ClockTimeProvider: public TimeProvider {
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    float startOffset;
    bool playing = false;

    void play();
    void pause();
    void seek(float time);

    bool isPlaying() const;

public:
    ClockTimeProvider();

    float getTime() const;
    void renderControls();

    void onFrameComplete();
};