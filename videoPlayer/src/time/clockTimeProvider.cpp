#include "clockTimeProvider.h"

#include <exception>
#include <iostream>

#include "imgui.h"

void ClockTimeProvider::play() {
    this->startTime = std::chrono::high_resolution_clock::now();
    this->playing = true;
}
void ClockTimeProvider::pause() {
    this->startOffset = this->getTime();
    this->playing = false;
}
void ClockTimeProvider::seek(float time) {
    if(time < 0.0f) {
        throw std::runtime_error("Cannot seek to negative time");
    }
    this->startOffset = time;
    if(this->playing) {
        this->startTime = std::chrono::high_resolution_clock::now();
    }
}

bool ClockTimeProvider::isPlaying() const{ 
    return this->playing; 
}

ClockTimeProvider::ClockTimeProvider() {
    this->startTime = std::chrono::high_resolution_clock::now();
}

float ClockTimeProvider::getTime() const {
    if(!playing) {
        return this->startOffset;
    }

    std::chrono::duration<float> offset = std::chrono::high_resolution_clock::now() - startTime;

    // TODO: use video end time
    float totalPlayPosition = offset.count() + this->startOffset;
    if(totalPlayPosition > 30.0f) {
        return 30.0f;
    }
    return totalPlayPosition;
}

void ClockTimeProvider::renderControls() {
    ImGui::Begin("Play controls");

    float originalTime = this->getTime();
    float time = originalTime;

    // TODO use video length
    ImGui::SliderFloat("Time", &time, 0.0f, 30.0f);
    if(time != originalTime) {
        std::cout << "Seeking " << abs(time-originalTime) << " seconds." << std::endl;
        this->seek(time);
    }

    if(this->isPlaying()) {
        if(ImGui::Button("Pause")) {
            this->pause();
        }
    } else {
        if(ImGui::Button("Play")) {
            this->play();
        }
    }
    if(ImGui::Button("Stop")) {
        if(this->isPlaying()) {
            this->pause();
        }
        this->seek(0.0f);
    }

    ImGui::End();
}


void ClockTimeProvider::onFrameComplete() {

}