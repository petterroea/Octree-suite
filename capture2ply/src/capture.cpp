#include "capture.h"

#include <rapidjson/document.h>

#include <json_helpers.h>

#include <depthCamera/drivers/realsenseDepthCamera.h>
#include <asyncPointcloudWriter.h>

#include <iostream>
#include <filesystem>
#include <chrono>

Capture::Capture(rapidjson::Document& document, std::filesystem::path workdir) {
    // Load the capture transform(Transform from world space to 1, -1 unit cube)
    assert(document.IsObject());
    assert(document.HasMember("capture_transform"));

    auto captureTransformArray = document["capture_transform"].GetArray();
    this->captureTransform = json_array_to_mat4x4(captureTransformArray);

    //Load cameras
    assert(document.HasMember("devices"));
    for(auto& device: document["devices"].GetArray()) {
        assert(device.HasMember("make"));
        assert(device.HasMember("serial"));

        assert(!strcmp(device["make"].GetString(), "realsense"));

        std::cout << " -> " << device["serial"].GetString() << std::endl;

        // Calculate capture path
        std::filesystem::path bagFilename(std::string("capture-") + device["serial"].GetString() + ".bag");
        std::filesystem::path fullFilename = workdir / bagFilename;

        if(!std::filesystem::exists(fullFilename)) {
            std::cout << fullFilename << " does not exist!" << std::endl;
            abort();
        }

        auto mode = RenderMode::HEADLESS;
        RealsenseDepthCamera* depthCamera = new RealsenseDepthCamera(RenderMode::HEADLESS, fullFilename.string(), device["serial"].GetString());
        this->cameras.push_back(depthCamera);
    }
}

Capture::~Capture() {
    for(auto camera: this->cameras) {
        delete camera;
    }
}

void Capture::to_ply(std::string outputDirectory) {
    std::cout << "Starting conversion to PLY frames" << std::endl;
    for(auto camera : this->cameras) {
        camera->beginStreaming();
    }

    auto pointcloudWriter = new AsyncPointcloudWriter(1920*1080*this->cameras.size());
    // Loop
    int framecount = 0;
    auto processing_start = std::chrono::system_clock::now();
    while(true) {
        std::cout << "---------------------------------" << std::endl;
        // Interface with realsense threads, fetching data
        bool outOfFrames = false;
        for(auto camera : this->cameras) {
            if(!camera->requestFrame()) {
                outOfFrames = true;
                break;
            }
        }
        if(outOfFrames) {
            break;
        }
        std::cout << "wait" << std::endl;
        for(auto camera : this->cameras) {
            camera->waitForNewFrame();
        }
        for(auto camera : this->cameras) {
            camera->uploadGpuDataSync();
        }

        auto pointcloud_generate_start = std::chrono::system_clock::now();
        // Generate pointcloud
        std::vector<Pointcloud> pointclouds;
        // If the thread isn't done writing the last thread(unlikely), wait
        pointcloudWriter->waitForSafeToWrite();
        // Transform points to world space and download them from the GPU
        for(auto camera : this->cameras) {
            Pointcloud pointcloud;
            camera->capturePoints(&pointcloud.points, &pointcloud.colors, &pointcloud.count, this->captureTransform);
            pointclouds.push_back(pointcloud);
        }
        pointcloudWriter->write(pointclouds);
        auto pointcloud_generate_end = std::chrono::system_clock::now();
        float elapsed_generate = std::chrono::duration_cast<std::chrono::milliseconds>(pointcloud_generate_end - pointcloud_generate_start).count();
        std::cout << "Generated pointcloud in " << elapsed_generate << "ms." << std::endl;
        framecount++;
    }
    std::cout << "One of the cameras ran out of frames, considering us done" << std::endl;
    // Request shutdown of the cameras
    for(auto camera : this->cameras) {
        camera->endCaptureThread();
    }
    for(auto camera : this->cameras) {
        camera->waitForThreadJoin();
    }

    auto processing_end = std::chrono::system_clock::now();
    // Cleanup
    float elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(processing_end- processing_start).count();
    float fps = static_cast<float>(framecount) / elapsed_time;

    delete pointcloudWriter;

    std::cout << "Processed " << framecount << " frames in " << elapsed_time << " seconds (" << fps << " fps)" << std::endl;
}