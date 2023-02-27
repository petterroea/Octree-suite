#include "cameraCalibrator.h"

bool CameraCalibrator::isEnabled() {
    return this->enabled;
}

void CameraCalibrator::setEnabled(bool newValue) {
    this->enabled = newValue;
}