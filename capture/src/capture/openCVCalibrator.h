#pragma once

#include <glm/mat4x4.hpp>
#include <librealsense2/rs.hpp>

#include <opencv2/core.hpp>
#include <opencv2/aruco.hpp>

#include <cameraCalibrator.h>

extern cv::Ptr<cv::aruco::Dictionary> openCVCalibrationDictionary;
extern cv::Ptr<cv::aruco::GridBoard> openCVCalibrationBoard;

/*
 * Responsible for storing OpenCV state for a depth camera
 * Is always owned by a depth camera
 * 
 * Currently mostly used to communicate state between a potential processing thread and the ImGui thread
 */
class OpenCVCalibrator : public CameraCalibrator {
    // Signal from processing thread
    int lastDetectedMarkers = 0;
    bool isValidPose = false;
    cv::Vec3d tvec;

    // Signal from GUI thread
    bool saveState = false; // not thread safe
public:
    OpenCVCalibrator();
    bool tryCalibrateCameraPosition(glm::mat4& transform, glm::mat3x3 cameraMatrixGlm, float* distCoeffs, int w, int h, void* data);

    void drawImmediateGui();
    int getLastDetectedMarkers() { return this->lastDetectedMarkers; }
    bool getIsValidPose() { return this->isValidPose; }
};