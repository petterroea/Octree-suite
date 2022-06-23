#include "depthCamera.h"

#include <imgui.h>

#include <librealsense2/hpp/rs_device.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include <iostream>

#include "calibration.h"

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
    ImGui::Checkbox("Enable OpenCV", &this->openCvEnabled);
    if(this->openCvEnabled) {
        // Align the color frame to the depth frame
        rs2::align align_to_depth(RS2_STREAM_DEPTH);
        current_frameset = align_to_depth.process(current_frameset);
        rs2::video_frame colorFrame = current_frameset.get_color_frame();
        tryCalibrateCameraPosition(this->calibratedTransform, colorFrame);
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
    //this->config.enable_all_streams();
    this->config.enable_stream(RS2_STREAM_COLOR, 1920, 1080, RS2_FORMAT_RGB8, 30);
    this->config.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 30);
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
