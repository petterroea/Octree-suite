#pragma once

#define RAPIDJSON_HAS_STDSTRING 1

#include <vector>
#include <chrono>
#include <string>

#include "../depthCamera/depthCamera.h"
#include "captureSettings.h"
#include "asyncPointcloudWriter.h"

class Capturer {
    std::vector<DepthCamera*> cameras;
    CaptureSettings settings;
    AsyncPointcloudWriter writer;

    bool autoCalibrate = false;
    int autoCalibrateTreshold = 45;

    void saveVideoMetadata(std::string filename);

    //Video-related stuff
    bool videoCapture = false;
    int framesCaptured = 0;
    std::chrono::time_point<std::chrono::system_clock> captureStart;

public:
    Capturer(std::vector<DepthCamera*> cameras);

    void getFrame();
    void render(glm::mat4x4& view, glm::mat4x4& proj);
    void displayGui();

    void saveCalibration(std::string filename);
    void loadCalibration();

    inline bool isCapturingVideo() { return this->videoCapture; };

    void capture();
};
