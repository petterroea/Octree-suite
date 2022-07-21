#include "realsenseDepthCamera.h"

#include <cudaHelpers.h>

#include <iostream>

#include "../../kernels/cudaPitchRgbToRgba.h"

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

    // Destination for RealSense RGB texture
    cudaMalloc(&this->cuTexRgb, this->videoMode.colorWidth*this->videoMode.colorHeight*3+1);
}

RealsenseDepthCamera::~RealsenseDepthCamera() {
    cudaFree(this->cuTexRgb);
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
        rs2::frameset alignedFrameset = align_to_depth.process(this->lastFrame);
        rs2::video_frame colorFrame = alignedFrameset.get_color_frame();
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
    // Upload color data
    cudaMemcpy(this->cuTexRgb, colorFrame.get_data(), w*h*3, cudaMemcpyHostToDevice);
    CUDA_CATCH_ERROR
    // Upload buffers
    cudaMemcpy(this->devPtrPoints, this->lastPointcloud.get_vertices(), sizeof(rs2::vertex)*this->lastPointcloud.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(this->devPtrTexCoords, this->lastPointcloud.get_texture_coordinates(), sizeof(rs2::texture_coordinate)*this->lastPointcloud.size(), cudaMemcpyHostToDevice);
    CUDA_CATCH_ERROR
    // Run our CUDA kernel that copies memory in parallel while expanding the pitch to RGBA
    pitchRgbToRgba(this->cuTexRgb, this->cuSurfaceObjectTexRgba, w*h, w);
    CUDA_CATCH_ERROR

    this->pointCount = lastPointcloud.size();

    auto end = std::chrono::system_clock::now();
    float elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    float processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(processing_end- start).count();
    float pointcloud_time = std::chrono::duration_cast<std::chrono::milliseconds>(pointcloud_end - processing_end).count();
    float gpu_upload_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - pointcloud_end).count();
    std::cout << this->getSerial() << " completed processing in " << elapsed_time << "ms (processing " << processing_time << " pointcloud " << pointcloud_time << ", gpu upload " << gpu_upload_time << ")" << std::endl;
}

void RealsenseDepthCamera::endCapture() {
    std::cout << this->getSerial() << " shutting down" << std::endl;
    this->capturePipeline.stop();
}


void RealsenseDepthCamera::uploadGpuDataSync() {
    // Everything is uploaded in CUDA
}