#include "realsenseDepthCamera.h"

#include <cudaHelpers.h>

#include <iostream>

RealsenseDepthCamera::RealsenseDepthCamera(RenderMode renderMode, rs2::device device, bool master) : DepthCamera(renderMode, VideoMode{
    .colorWidth = 1920,
    .colorHeight = 1080,
    .depthWidth = 1280,
    .depthHeight = 720
}), device(device){
    /*
     * Forces hardware sync
     * See https://dev.intelrealsense.com/docs/multiple-depth-cameras-configuration#section-g-hw-sync-validation
     * And https://github.com/IntelRealSense/librealsense/issues/8529
    */
    auto depthSensor = device.first<rs2::depth_sensor>();
    depthSensor.set_option(RS2_OPTION_INTER_CAM_SYNC_MODE, master ? 1 : 2);

    // Force the same settings everywhere
    // Disable auto exposure because it means cameras have different color balances
    //depthSensor.set_option(RS2_OPTION_AVALANCHE_PHOTO_DIODE, 33000.0f);
    //depthSensor.set_option(RS2_OPTION_DIGITAL_GAIN, 16.0f);
    depthSensor.set_option(RS2_OPTION_LASER_POWER, 150.0f);

    auto colorSensor = device.first<rs2::color_sensor>();
    colorSensor.set_option(RS2_OPTION_EXPOSURE, 166.0f);
    colorSensor.set_option(RS2_OPTION_GAIN, 60.0f);
    colorSensor.set_option(RS2_OPTION_WHITE_BALANCE, 4000.0f);

    colorSensor.set_option(RS2_OPTION_BRIGHTNESS, 0.0f);
    colorSensor.set_option(RS2_OPTION_CONTRAST, 50.0f);
    colorSensor.set_option(RS2_OPTION_GAMMA, 300.0f);
    colorSensor.set_option(RS2_OPTION_HUE, 0.0f);
    colorSensor.set_option(RS2_OPTION_SATURATION, 64.0f);
    colorSensor.set_option(RS2_OPTION_SHARPNESS, 50.0f);

    this->textureConversionBuffer = new unsigned char[1920*4*1080];
}

RealsenseDepthCamera::~RealsenseDepthCamera() {
    delete[] this->textureConversionBuffer;
}

std::string RealsenseDepthCamera::getSerial() { 
    return this->device.get_info(rs2_camera_info::RS2_CAMERA_INFO_SERIAL_NUMBER);
}

std::string RealsenseDepthCamera::getKind() { 
    return "RealSense Camera";
}

void RealsenseDepthCamera::beginCapture() {
    this->config.enable_device(this->device.get_info(rs2_camera_info::RS2_CAMERA_INFO_SERIAL_NUMBER));
    //this->config.enable_all_streams();
    this->config.enable_stream(RS2_STREAM_COLOR, 1920, 1080, RS2_FORMAT_RGB8, 30);
    this->config.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 30);

    this->capturePipeline.start(this->config);
}

void RealsenseDepthCamera::processFrame() {
    auto start = std::chrono::system_clock::now();
    std::cout << this->getSerial() << "Frame requested" << std::endl;

    // Wait for a frame from realSense
    this->lastFrame = this->capturePipeline.wait_for_frames();

    if(this->calibrationEnabled) {
        // Align the color frame to the depth frame so OpenCV gets correct data
        rs2::align align_to_depth(RS2_STREAM_DEPTH);
        this->lastFrame = align_to_depth.process(this->lastFrame);
        rs2::video_frame colorFrame = this->lastFrame.get_color_frame();
        this->calibrator.tryCalibrateCameraPosition(this->calibratedTransform, colorFrame);
    }
    std::cout << this->getSerial() << " processed a frame" << std::endl;
    auto processing_end = std::chrono::system_clock::now();
    rs2::video_frame colorFrame = this->lastFrame.get_color_frame();

    // Generate pointcloud
    this->pointcloud.map_to(colorFrame);
    this->lastPointcloud = this->pointcloud.calculate(this->lastFrame.get_depth_frame());
    auto pointcloud_end = std::chrono::system_clock::now();

    int w = colorFrame.get_width();
    int h = colorFrame.get_height();
    unsigned char* data = (unsigned char*) colorFrame.get_data();
    for(int i = 0; i < w*h; i++) {
        this->textureConversionBuffer[i*4+0] = data[i*3+0];
        this->textureConversionBuffer[i*4+1] = data[i*3+1];
        this->textureConversionBuffer[i*4+2] = data[i*3+2];
        this->textureConversionBuffer[i*4+3] = 1;
    }
    // Upload color data
    cudaMemcpy2DToArray(
        this->cuArrayTexRgb, 0, 0, 
        textureConversionBuffer, 
        // Pitch
        colorFrame.get_width()*4, 
        // Width
        colorFrame.get_width()*4, 
        colorFrame.get_height(), 
        cudaMemcpyHostToDevice
    );
    CUDA_CATCH_ERROR

    auto end = std::chrono::system_clock::now();
    float elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    float processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(processing_end- start).count();
    float pointcloud_time = std::chrono::duration_cast<std::chrono::milliseconds>(pointcloud_end - processing_end).count();
    std::cout << this->getSerial() << " completed processing in " << elapsed_time << "ms (processing " << processing_time << " pointcloud " << pointcloud_time << ")" << std::endl;
}

void RealsenseDepthCamera::endCapture() {
    std::cout << this->getSerial() << " shutting down" << std::endl;
    this->capturePipeline.stop();
}


void RealsenseDepthCamera::uploadGpuDataSync() {
    std::cout << this->getSerial() << " uploading textures" << std::endl;
    if(renderMode == RenderMode::OPENGL) {
        // TODO can we upload from cuda once we have uploaded from OpenGL once?
        glBindTexture(GL_TEXTURE_2D, this->renderer->getDepthTextureHandle());
        rs2::depth_frame depth = this->lastFrame.get_depth_frame();
        std::cout << "Depth: " << depth.get_width() << " " << depth.get_height() << std::endl;
        //glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, depth.get_width(), depth.get_height(), 0, GL_RED, GL_UNSIGNED_SHORT, depth.get_data());

        glBindTexture(GL_TEXTURE_2D, this->renderer->getColorTextureHandle());
        rs2::video_frame colorFrame = this->lastFrame.get_color_frame();
        std::cout << "Color: " << colorFrame.get_width() << " " << colorFrame.get_height() << std::endl;
        //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, colorFrame.get_width(), colorFrame.get_height(), 0, GL_RGB, GL_UNSIGNED_BYTE, colorFrame.get_data());

        glBindTexture(GL_TEXTURE_2D, 0);
        std::cout << this->getSerial() << " uploading buffers" << std::endl;
        glBindBuffer(GL_ARRAY_BUFFER, this->renderer->getPointBufferHandle());
        glBufferData(GL_ARRAY_BUFFER, sizeof(rs2::vertex)*lastPointcloud.size(), lastPointcloud.get_vertices(), GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, this->renderer->getTextureCoordBufferHandle());
        glBufferData(GL_ARRAY_BUFFER, sizeof(rs2::texture_coordinate)*lastPointcloud.size(), lastPointcloud.get_texture_coordinates(), GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    } else {
        throw "TODO";
    }

    this->pointCount = lastPointcloud.size();
}