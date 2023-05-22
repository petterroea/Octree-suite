#include "recordingTimeProvider.h"

#include <iostream>

RecordingTimeProvider::RecordingTimeProvider(float fps, int frameCount) : fps(fps), frameCount(frameCount) {

}

float RecordingTimeProvider::getTime() const {
    float time = static_cast<float>(this->frameCounter) / this->fps;
    return time;
}

int RecordingTimeProvider::getCurrentFrame() const {
    return this->frameCounter;
}

void RecordingTimeProvider::renderControls() {
    // Do nothing
}

void RecordingTimeProvider::onFrameComplete() {
    this->frameCounter++;
    float time = this->getTime();
    float totalTime = static_cast<float>(frameCount) / this->fps;
    if(time > totalTime) {
        std::cout << "Detected end of video, quitting..." << std::endl;
        exit(0);
    } else {
        std::cout << "Render progress: " << time << " out of " << totalTime << " seconds. ( " << ((time/totalTime)*100.0f) << "% )" << std::endl;
    }
}