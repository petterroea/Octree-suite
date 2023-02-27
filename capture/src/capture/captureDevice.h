#pragma once

#include "openCVCalibrator.h"
#include <depthCamera/depthCamera.h>

class CaptureDevice {
    DepthCamera* camera;

    bool calibrationEnabled = false;
    OpenCVCalibrator* calibrator;

public:
    CaptureDevice(DepthCamera* camera, OpenCVCalibrator* calibrator);
    ~CaptureDevice();

    DepthCamera* getDepthCamera();

    OpenCVCalibrator* getCalibrator();

    void drawImmediateGui();
};