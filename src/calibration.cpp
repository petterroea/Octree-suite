#include "calibration.h"

#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/calib3d.hpp>

#include <imgui.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

//Derived by means of experimentation
//glm::vec3 sensorOffset(0.010f, 0.002f, -0.016f);
// Let's not care about sensor offset for now
glm::vec3 sensorOffset(0.0f, 0.0f, 0.0f);

// Just store these as static variables
// Some global truths about the arucoboard we are using
cv::Ptr<cv::aruco::Dictionary> openCVCalibrationDictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
cv::Ptr<cv::aruco::GridBoard> openCVCalibrationBoard = cv::aruco::GridBoard::create(6, 8, 0.027, 0.0065, openCVCalibrationDictionary);

bool tryCalibrateCameraPosition(glm::mat4& transform, rs2::video_frame& frame) {
    bool success = false;
    cv::Mat image = cv::Mat(frame.get_width()*frame.get_height(), 1, CV_8UC3, (void*)frame.get_data()).clone().reshape(0, frame.get_height());
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f> > corners;
    cv::aruco::detectMarkers(image, openCVCalibrationDictionary, corners, ids);
    ImGui::Text("Detected markers: %ld", ids.size());

    //Get camera intrinsics
    rs2_intrinsics intrinsics = frame.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
    if(ids.size() > 0) {
        cv::Mat cameraMatrix = (cv::Mat_<float>(3,3) << 
        intrinsics.fx, 0.0f, intrinsics.ppx, 
        0.0f, intrinsics.fy, intrinsics.ppy,
        0.0f, 0.0f, 1.0f);
        cv::Mat distCoeffs = (cv::Mat_<float>(5,1) << 
            intrinsics.coeffs[0],
            intrinsics.coeffs[1],
            intrinsics.coeffs[2],
            intrinsics.coeffs[3],
            intrinsics.coeffs[4]
        );
        cv::Vec3d rvec, tvec;
        int valid = cv::aruco::estimatePoseBoard(corners, ids, openCVCalibrationBoard, cameraMatrix, distCoeffs, rvec, tvec);
        if(valid > 0)
            cv::drawFrameAxes(image, cameraMatrix, distCoeffs, rvec, tvec, 0.1);

        ImGui::Text("Valid pose?: %s", valid > 0 ? "Yes" : "No");
        if(valid > 0) {
            ImGui::Text("tvec: %f %f %f", tvec[0], tvec[1], tvec[2]);
            if(true || ImGui::Button("Calibrate camera")) {
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

                ImGui::SliderFloat3("Sensor offset", (float*)&sensorOffset, -0.5f, 0.5f);
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
    }

    if(ImGui::Button("Save me some state, please")) {
        // Draw the detected markers
        if(ids.size() > 0) {
            cv::aruco::drawDetectedMarkers(image, corners, ids);
        }
        cv::imwrite("state.png", image);
    }
    return success;
}