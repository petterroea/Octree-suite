#include "captureDevice.h"

#include <imgui.h>

CaptureDevice::CaptureDevice(DepthCamera* camera, OpenCVCalibrator* calibrator) : camera(camera), calibrator(calibrator) {

}

CaptureDevice::~CaptureDevice() {

}

DepthCamera* CaptureDevice::getDepthCamera() {
    return this->camera;
}

OpenCVCalibrator* CaptureDevice::getCalibrator() {
    return this->calibrator;
}

bool CaptureDevice::isCalibrationEnabled() {
    return this->calibrationEnabled;
}

void CaptureDevice::setCalibrationEnabled(bool newState) {
    this->calibrationEnabled = newState;
}

void CaptureDevice::drawImmediateGui() {
    ImGui::Checkbox("Enable OpenCV", &this->calibrationEnabled);
    this->calibrator->drawImmediateGui();
}