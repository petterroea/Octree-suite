#include "openCVCalibrator.h"

#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <imgui.h>
#include <iostream>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

//Derived by means of experimentation
//glm::vec3 sensorOffset(0.010f, 0.002f, -0.016f);
// Let's not care about sensor offset for now
glm::vec3 sensorOffset(0.0f, 0.0f, 0.0f);

// Just store these as static variables
// Some global truths about the arucoboard we are using
cv::Ptr<cv::aruco::Dictionary> openCVCalibrationDictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
cv::Ptr<cv::aruco::GridBoard> openCVCalibrationBoard = cv::aruco::GridBoard::create(6, 8, 4.0f/100.0f, 0.95/100.0f, openCVCalibrationDictionary);

bool OpenCVCalibrator::tryCalibrateCameraPosition(glm::mat4& transform, glm::mat3x3 cameraMatrixGlm, float* distCoeffs , int w, int h, void* data) {
    std::cout << "CALIBRATING" << std::endl;
    bool success = false;
    cv::Mat image = cv::Mat(w*h, 1, CV_8UC3, data).clone().reshape(0, h);
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f> > corners;
    cv::aruco::detectMarkers(image, openCVCalibrationDictionary, corners, ids);
    this->lastDetectedMarkers = ids.size();

    //Get camera intrinsics
    if(ids.size() > 0) {
        cv::Mat cameraMatrix = (cv::Mat_<float>(3,3) << 
            cameraMatrixGlm[0][0], cameraMatrixGlm[0][1], cameraMatrixGlm[0][2],
            cameraMatrixGlm[1][0], cameraMatrixGlm[1][1], cameraMatrixGlm[1][2],
            cameraMatrixGlm[2][0], cameraMatrixGlm[2][1], cameraMatrixGlm[2][2]
        );
        cv::Mat distCoeffs_cv = (cv::Mat_<float>(5,1) << 
            distCoeffs[0],
            distCoeffs[1],
            distCoeffs[2],
            distCoeffs[3],
            distCoeffs[4]
        );
        cv::Vec3d rvec;
        int valid = cv::aruco::estimatePoseBoard(corners, ids, openCVCalibrationBoard, cameraMatrix, distCoeffs_cv, rvec, tvec);
        this->isValidPose = valid > 0;

        if(valid > 0)
            cv::drawFrameAxes(image, cameraMatrix, distCoeffs_cv, rvec, tvec, 0.1);

        if(valid > 0) {
            float x = rvec[0];
            float y = rvec[1];
            float z = rvec[2];
            float len = cv::norm(rvec);
            glm::mat4 rotation = glm::rotate(
                len,
                glm::normalize(glm::vec3(x, y, z))
            );
            glm::mat4 translation = glm::translate(
                glm::mat4(1.0f),
                //Hack
                glm::vec3(tvec[0], tvec[1], tvec[2])
            );

            glm::mat4 sensorTranslation = glm::translate(
                glm::mat4(1.0f),
                sensorOffset
            );
            transform = 
                glm::inverse(rotation) *
                glm::inverse(translation) *
                sensorTranslation;
            success = true;
        }
    }

    if(this->saveState) {
        this->saveState = false;
        // Draw the detected markers
        if(ids.size() > 0) {
            cv::aruco::drawDetectedMarkers(image, corners, ids);
        }
        cv::imwrite("state.png", image);
    }
    return success;
}

void OpenCVCalibrator::drawImmediateGui() {
    ImGui::Text("Detected markers: %d", this->lastDetectedMarkers);
    ImGui::Text("Valid pose?: %s", isValidPose ? "Yes" : "No");
    if(isValidPose) {
        ImGui::Text("tvec: %f %f %f", tvec[0], tvec[1], tvec[2]);
    }
    ImGui::SliderFloat3("Sensor offset", (float*)&sensorOffset, -0.5f, 0.5f);
    if(ImGui::Button("Save me some state, please")) {
        this->saveState = true;
    }
}

OpenCVCalibrator::OpenCVCalibrator() {

}