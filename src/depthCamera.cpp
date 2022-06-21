#include "depthCamera.h"

#include <imgui.h>

#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/calib3d.hpp>

#include <librealsense2/hpp/rs_device.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include <iostream>

GLuint DepthCamera::buildTexture() {
    // Create a OpenGL texture identifier
    GLuint image_texture;
    glGenTextures(1, &image_texture);
    std::cout << "Got " << image_texture << std::endl;
    glBindTexture(GL_TEXTURE_2D, image_texture);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same

    // Upload pixels into texture
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data);
    glBindTexture(GL_TEXTURE_2D, 0);
    return image_texture;
}

rs2::frameset DepthCamera::processFrame() {
    rs2::frameset current_frameset = this->capturePipeline.wait_for_frames();
    ImGui::Checkbox(("Enable OpenCV for " + this->getSerial()).c_str(), &this->openCvEnabled);
    if(this->openCvEnabled) {
        rs2::video_frame colorFrame = current_frameset.get_color_frame();
        cv::Mat image = cv::Mat(colorFrame.get_width()*colorFrame.get_height(), 1, CV_8UC3, (void*)colorFrame.get_data()).clone().reshape(0, colorFrame.get_height());
        cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f> > corners;
        cv::aruco::detectMarkers(image, dictionary, corners, ids);
        ImGui::Text("Detected markers: %ld", ids.size());

        //Get camera intrinsics
        rs2_intrinsics intrinsics = colorFrame.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
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
            cv::Ptr<cv::aruco::GridBoard> board = cv::aruco::GridBoard::create(6, 8, 0.027, 0.0065, dictionary);
            int valid = cv::aruco::estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs, rvec, tvec);
            if(valid > 0)
                cv::drawFrameAxes(image, cameraMatrix, distCoeffs, rvec, tvec, 0.1);

            ImGui::Text("Valid pose?: %s", valid > 0 ? "Yes" : "No");
            if(valid > 0) {
                ImGui::Text("tvec: %f %f %f", tvec[0], tvec[1], tvec[2]);
                if(true || ImGui::Button(("Calibrate camera" + this->getSerial()).c_str())) {
                    //cv::Mat rmat;
                    //cv::Rodrigues(rvec, rmat);
                    /*
                    glm::mat4 rotation = glm::mat4(
                        rmat.at<float>(0, 0), rmat.at<float>(1, 0), rmat.at<float>(2, 0), 0.0f,
                        rmat.at<float>(0, 1), rmat.at<float>(1, 1), rmat.at<float>(2, 1), 0.0f,
                        rmat.at<float>(0, 2), rmat.at<float>(1, 2), rmat.at<float>(2, 2), 0.0f,
                        0.0f, 0.0f, 0.0f, 1.0f
                    );
                    */
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
                        glm::vec3(tvec[0]-0.0425f, tvec[1], tvec[2])
                    );

                    //rotation = glm::transpose(rotation);

                    this->calibratedTransform = 
                        glm::inverse(rotation) *
                        glm::inverse(translation);
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
    }
    return current_frameset;
}

void DepthCamera::uploadTextures(rs2::frameset& frame) {
    // Try to get a frame of a depth image
    glBindTexture(GL_TEXTURE_2D, this->depthTexture);
    rs2::depth_frame depth = frame.get_depth_frame();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, depth.get_width(), depth.get_height(), 0, GL_RED, GL_UNSIGNED_SHORT, depth.get_data());

    glBindTexture(GL_TEXTURE_2D, this->colorTexture);
    rs2::video_frame colorFrame = frame.get_color_frame();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, colorFrame.get_width(), colorFrame.get_height(), 0, GL_RGB, GL_UNSIGNED_BYTE, colorFrame.get_data());

    glBindTexture(GL_TEXTURE_2D, 0);
}

rs2::points DepthCamera::processPointcloud(rs2::frameset& frame) {
    this->pointcloud.map_to(frame.get_color_frame());
    rs2::points points = this->pointcloud.calculate(frame.get_depth_frame());
    return points;
}
/*
void DepthCamera::depthCameraThread(DepthCamera* camera) {
    // Create a dedicated pipeline and enable it

    while(camera->shouldRunThread) {
        // Update textures using realsense data

        camera->current_frameset = current;
    }
}
*/

DepthCamera::DepthCamera(rs2::device device): device(device), calibratedTransform(1.0f) {
    // Buffer to use to display from camera
    this->colorTexture = DepthCamera::buildTexture();
    this->depthTexture = DepthCamera::buildTexture();

    std::cout << "Color texture " << this->colorTexture << std::endl;
    std::cout << "Depth texture " << this->depthTexture << std::endl;

    this->config.enable_device(this->device.get_info(rs2_camera_info::RS2_CAMERA_INFO_SERIAL_NUMBER));
    this->config.enable_all_streams();
}

DepthCamera::~DepthCamera() {
    glDeleteTextures(1, &this->colorTexture);
    glDeleteTextures(1, &this->depthTexture);
}

void DepthCamera::begin() {
    this->capturePipeline.start(this->config);
}

void DepthCamera::end() {
    this->capturePipeline.stop();
}
