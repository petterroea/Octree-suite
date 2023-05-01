#pragma once

class TimeProvider {
public:
    virtual float getTime() const = 0;
    virtual void renderControls() = 0;

    virtual void onFrameComplete() = 0;
};