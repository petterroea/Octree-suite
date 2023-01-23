#pragma once

class CameraCalibrator {

public:
    virtual bool isEnabled() {};
    // Called from the camera
    virtual bool tryClaibrateCameraPosition(int w, int h, void* data) {};
};