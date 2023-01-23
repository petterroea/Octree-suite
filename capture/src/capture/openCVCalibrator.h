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
    bool tryCalibrateCameraPosition(glm::mat4& transform, rs2::video_frame& frame);

    void drawImmediateGui();
    int getLastDetectedMarkers() { return this->lastDetectedMarkers; }
    bool getIsValidPose() { return this->isValidPose; }
};