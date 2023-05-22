#pragma once

#include <glm/mat4x4.hpp>

class CameraCalibrator {
    bool enabled;
public:
    bool isEnabled();
    void setEnabled(bool newValue);
    // Called from the camera
    virtual bool tryCalibrateCameraPosition(glm::mat4& transform, glm::mat3x3 cameraMatrix, float* distCoeffs, int w, int h, void* data) = 0;
};