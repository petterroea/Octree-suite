#include "depthCamera.h"

#include <imgui.h>

#include <librealsense2/hpp/rs_device.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include <iostream>

#include "openCVCalibrator.h"

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

void DepthCamera::processFrame() {
    this->lastFrame = this->capturePipeline.wait_for_frames();
    if(this->calibrationEnabled) {
        // Align the color frame to the depth frame
        rs2::align align_to_depth(RS2_STREAM_DEPTH);
        this->lastFrame = align_to_depth.process(this->lastFrame);
        rs2::video_frame colorFrame = this->lastFrame.get_color_frame();
        this->calibrator.tryCalibrateCameraPosition(this->calibratedTransform, colorFrame);
    }
    std::cout << this->getSerial() << " processed a frame" << std::endl;
}

void DepthCamera::uploadTextures() {
    std::cout << this->getSerial() << " uploading textures" << std::endl;
    // Try to get a frame of a depth image
    glBindTexture(GL_TEXTURE_2D, this->depthTexture);
    rs2::depth_frame depth = this->lastFrame.get_depth_frame();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, depth.get_width(), depth.get_height(), 0, GL_RED, GL_UNSIGNED_SHORT, depth.get_data());

    glBindTexture(GL_TEXTURE_2D, this->colorTexture);
    rs2::video_frame colorFrame = this->lastFrame.get_color_frame();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, colorFrame.get_width(), colorFrame.get_height(), 0, GL_RGB, GL_UNSIGNED_BYTE, colorFrame.get_data());

    glBindTexture(GL_TEXTURE_2D, 0);
}

rs2::points DepthCamera::getLastPointcloud() {
    return this->lastPointcloud;
}

rs2::points DepthCamera::processPointcloud(rs2::frameset& frame) {
    this->pointcloud.map_to(frame.get_color_frame());
    rs2::points points = this->pointcloud.calculate(frame.get_depth_frame());
    return points;
}

void DepthCamera::drawImmediateGui() {
    ImGui::Checkbox("Enable OpenCV", &this->calibrationEnabled);

    this->calibrator.drawImmediateGui();
}

DepthCamera::DepthCamera(rs2::device device, bool master): device(device), calibratedTransform(1.0f) {
    // Buffer to use to display from camera
    this->colorTexture = DepthCamera::buildTexture();
    this->depthTexture = DepthCamera::buildTexture();

    /*
     * Forces hardware sync
     * See https://dev.intelrealsense.com/docs/multiple-depth-cameras-configuration#section-g-hw-sync-validation
     * And https://github.com/IntelRealSense/librealsense/issues/8529
    */
    device.first<rs2::depth_sensor>().set_option(RS2_OPTION_INTER_CAM_SYNC_MODE, master ? 1 : 2);

    std::cout << "Color texture " << this->colorTexture << std::endl;
    std::cout << "Depth texture " << this->depthTexture << std::endl;
}

DepthCamera::~DepthCamera() {
    glDeleteTextures(1, &this->colorTexture);
    glDeleteTextures(1, &this->depthTexture);
}

void DepthCamera::begin() {
    sem_init(&this->frameRequestSemaphore, 0, 0);
    sem_init(&this->frameReceivedSemaphore, 0, 0);
    this->hThread = pthread_create(&this->hThread, NULL, (void* (*)(void*))DepthCamera::threadEntrypoint, (void*)this);
}

void DepthCamera::end() {
    this->running = false;
    // Make sure the loop gets to finish
    this->requestFrame();
}

void DepthCamera::waitForThreadJoin() {
    pthread_join(this->hThread, NULL);
}

void DepthCamera::threadEntrypoint(DepthCamera* self) {
    self->processingThread();
}

void DepthCamera::processingThread() {
    this->config.enable_device(this->device.get_info(rs2_camera_info::RS2_CAMERA_INFO_SERIAL_NUMBER));
    //this->config.enable_all_streams();
    this->config.enable_stream(RS2_STREAM_COLOR, 1920, 1080, RS2_FORMAT_RGB8, 30);
    this->config.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 30);

    this->capturePipeline.start(this->config);
    while(this->running) {
        sem_wait(&this->frameRequestSemaphore);

        auto start = std::chrono::system_clock::now();
        std::cout << this->getSerial() << "Frame requested" << std::endl;

        this->processFrame();
        auto processing_end = std::chrono::system_clock::now();
        this->lastPointcloud = this->processPointcloud(lastFrame);
        auto pointcloud_end = std::chrono::system_clock::now();

        auto end = std::chrono::system_clock::now();
        float elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        float processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(processing_end- start).count();
        float pointcloud_time = std::chrono::duration_cast<std::chrono::milliseconds>(pointcloud_end - processing_end).count();
        std::cout << this->getSerial() << " completed processing in " << elapsed_time << "ms (processing " << processing_time << " pointcloud " << pointcloud_time << ")" << std::endl;

        sem_post(&this->frameReceivedSemaphore);
    }
    std::cout << this->getSerial() << " shutting down" << std::endl;
    this->capturePipeline.stop();
}

void DepthCamera::requestFrame() {
    std::cout << this->getSerial() << " Requesting frame" << std::endl;
    sem_post(&this->frameRequestSemaphore);
}
void DepthCamera::waitForNewFrame() {
    sem_wait(&this->frameReceivedSemaphore);
}