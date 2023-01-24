#include "capturer.h"

#include <utils.h>

#include <imgui.h>

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

#include <rapidjson/document.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/prettywriter.h>

#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <fstream>
#include <filesystem>
#include <stdexcept>

#include <json_helpers.h>

namespace fs = std::filesystem;

Capturer::Capturer(std::vector<CaptureDevice*> devices) : writer(1920*1080*devices.size()) {
    this->captureDevices = devices;
}
Capturer::~Capturer() {
    for(auto device : this->captureDevices) {
        delete device->getDepthCamera();
    }
}

void Capturer::getFrame() {
    std::cout << "- getFrame called" << std::endl;
    for(auto device : this->captureDevices) {
        if(this->autoCalibrate) {
            device->setCalibrationEnabled(true);
        }
        device->getDepthCamera()->requestFrame();
    }
    for(auto device : this->captureDevices) {
        device->getDepthCamera()->waitForNewFrame();
    }
    if(this->autoCalibrate) {
        // Check if all cameras see enough markers
        bool allGood = true;
        for(auto device : this->captureDevices) {
            if(!device->getCalibrator()->getIsValidPose() || device->getCalibrator()->getLastDetectedMarkers() < this->autoCalibrateTreshold) {
                allGood = false;
            }
        }
        if(allGood) {
            this->autoCalibrate = false;

            for(auto device : this->captureDevices) {
                device->setCalibrationEnabled(false);
            }
        }
    }
    //Optional
    for(auto device : this->captureDevices) {
        device->getDepthCamera()->uploadGpuDataSync();
    }
}

void Capturer::saveCalibration(std::string filename) {
    rapidjson::Document d;
    d.SetObject();
    rapidjson::Document::AllocatorType& allocator = d.GetAllocator();
    for(auto device : this->captureDevices) {
        glm::mat4x4 matrix = device->getDepthCamera()->getCalibration();
        rapidjson::Value deviceMatrixArray = mat4x4_to_json_array(matrix, allocator);

        rapidjson::Value key(device->getDepthCamera()->getSerial().c_str(), allocator); // copy string name
        d.AddMember(key, deviceMatrixArray, allocator);
    }

    std::ofstream out(filename);
    rapidjson::OStreamWrapper osw(out);

    rapidjson::PrettyWriter<rapidjson::OStreamWrapper> writer(osw);
    d.Accept(writer);
}

void Capturer::loadCalibration() {
    std::ifstream ifs("calibration.json");
    rapidjson::IStreamWrapper isw(ifs);
    
    rapidjson::Document d;
    d.ParseStream(isw);

    bool error = false;
    bool incomplete = false;

    for(auto device : this->captureDevices) {
        auto depthCamera = device->getDepthCamera();
        rapidjson::Value::ConstMemberIterator itr = d.FindMember(depthCamera->getSerial().c_str());
        if (itr != d.MemberEnd()) {
            rapidjson::Value& a = d[depthCamera->getSerial().c_str()];
            try {
                glm::mat4x4 calibration = json_array_to_mat4x4(a);
                depthCamera->setCalibration(calibration);
            } catch(std::invalid_argument const& ex) {
                error = true;
                ImGui::OpenPopup("Error loading calibration");
                std::cout << "Unable to load calibration: " << ex.what() << std::endl;
                break;
            }
        } else {
            incomplete = true;
        }
    }
}

void Capturer::render(glm::mat4x4& view, glm::mat4x4& proj) {
    for(auto device : this->captureDevices) {
        auto depthCamera = device->getDepthCamera();
        depthCamera->getRenderer()->render(depthCamera->getCalibration(), view, proj, depthCamera->getPointCount());
    }
    this->settings.renderHelpLines(view, proj);

    if(this->videoCapture) {
        this->framesCaptured++;
    }
}
glm::mat4x4 Capturer::getCaptureTransform() {
    return glm::scale(
        glm::translate(
            glm::mat4(1.0f), 
            -1.0f*this->settings.getCapturePosition()), 
        glm::vec3(1.0f/this->settings.getCaptureScale())
    );
}

void Capturer::capture() {
    std::vector<Pointcloud> pointclouds;
    // If the thread isn't done writing the last thread(unlikely), wait
    this->writer.waitForSafeToWrite();
    // Calculate the transformation matrix for the configured capture region
    glm::mat4x4 captureTransform = this->getCaptureTransform();
    // Transform points to world space and download them from the GPU
    for(auto device : this->captureDevices) {
        Pointcloud pointcloud;
        device->getDepthCamera()->capturePoints(&pointcloud.points, &pointcloud.colors, &pointcloud.count, captureTransform);
        pointclouds.push_back(pointcloud);
    }
    writer.write(pointclouds);
}

void Capturer::displayGui() {
    ImGui::Begin("Capture");
    this->settings.displayGui();
    if(ImGui::Button("Capture pointcloud")) {
        this->capture();
    }
    if(this->videoCapture) {
        if(ImGui::Button("Stop capture")) {
            std::cout << "Ending video capture" << std::endl;
            this->videoCapture = false;
            this->framesCaptured = 0;
            for(auto device : this->captureDevices) {
                device->getDepthCamera()->endCaptureThread();
            }
            std::cout << "Sent stop signal to threads" << std::endl;
            for(auto device : this->captureDevices) {
                device->getDepthCamera()->waitForThreadJoin();
            }
            std::cout << "Threads joined" << std::endl;
            for(auto device : this->captureDevices) {
                device->getDepthCamera()->beginStreaming();
            }
            std::cout << "Streaming started" << std::endl;
        }
        std::chrono::duration<float> elapsedTime = (std::chrono::system_clock::now() - this->captureStart);
        float fps = static_cast<float>(this->framesCaptured) / elapsedTime.count();
        ImGui::Text("%d frames in %lf seconds(%f FPS)", this->framesCaptured, elapsedTime.count(), fps);
    } else {
        if(ImGui::Button("Capture video")) {
            std::cout << "Starting video capture" << std::endl;

            auto time_str = currentTimeAsString();

            //Create a directory for the capture
            auto capture_directory = "./captures/" + time_str;
            fs::create_directories(capture_directory);
            this->saveVideoMetadata(capture_directory + "/metadata.json");

            this->videoCapture = true;
            this->captureStart = std::chrono::system_clock::now();
            for(auto device : this->captureDevices) {
                device->getDepthCamera()->endCaptureThread();
            }
            for(auto device : this->captureDevices) {
                device->getDepthCamera()->waitForThreadJoin();
            }
            std::cout << "Threads joined" << std::endl;
            for(auto device: this->captureDevices) {
                auto depthCamera = device->getDepthCamera();
                depthCamera->beginRecording(capture_directory + "/capture-" + depthCamera->getSerial() + ".bag");
            }
        }
    }
    ImGui::End();
    ImGui::Begin("Devices");
    for(auto device : this->captureDevices) {
        auto camera = device->getDepthCamera();
        ImGui::PushID(camera->getSerial().c_str());

        char label[200];
        sprintf(label, "%s: %s", camera->getKind().c_str(), camera->getSerial().c_str());
        if(ImGui::CollapsingHeader(label)) {
            device->drawImmediateGui();
            ImGui::Separator();
        }
        ImGui::PopID();
    }
    ImGui::End();

    ImGui::Begin("Utils");
    if(ImGui::Button("Save calibration to JSON")) {
        saveCalibration("calibration.json");
    }
    ImGui::SameLine();
    if(ImGui::Button("Load calibration from JSON")) {
        loadCalibration();
    }
    if(ImGui::CollapsingHeader("OpenCV")) {
        ImGui::Checkbox("Autocalibrate", &this->autoCalibrate);
        ImGui::SliderInt("Autocalibrate treshold", &this->autoCalibrateTreshold, 5, 6*8);

        if(ImGui::Button("Generate ArUco board")) {
            cv::Mat boardImage;
            openCVCalibrationBoard->draw( cv::Size(794*4, 1123*4), boardImage, 1, 1 );
            cv::imwrite("ArUco.bmp", boardImage);
        }
    }
    ImGui::End();
}

void Capturer::saveVideoMetadata(std::string filename) {
    rapidjson::Document d;
    d.SetObject();
    rapidjson::Document::AllocatorType& allocator = d.GetAllocator();
    {
        rapidjson::Value version;
        version.SetString("1");

        rapidjson::Value key;
        key.SetString("version");
        d.AddMember(key, version, allocator);
    }
    {
        glm::mat4x4 capture_transform = this->getCaptureTransform();
        rapidjson::Value capture_transform_json = mat4x4_to_json_array(capture_transform, allocator);

        rapidjson::Value key;
        key.SetString("capture_transform");
        d.AddMember(key, capture_transform_json, allocator);
    }
    {
        rapidjson::Value devices(rapidjson::Type::kArrayType);

        for(auto device: this->captureDevices) {
            rapidjson::Value deviceJsonObj(rapidjson::Type::kObjectType);
            // Camera serial
            {
                rapidjson::Value cameraSerialValue;
                auto cameraSerial = device->getDepthCamera()->getSerial();
                cameraSerialValue.SetString(cameraSerial.c_str(), cameraSerial.length(), allocator);

                rapidjson::Value serialKey;
                serialKey.SetString("serial");

                deviceJsonObj.AddMember(serialKey, cameraSerial, allocator);
            }
            // Camera make
            {
                rapidjson::Value cameraMake;
                cameraMake.SetString("realsense");

                rapidjson::Value makeKey;
                makeKey.SetString("make");
                deviceJsonObj.AddMember(makeKey, cameraMake, allocator);

            }
            // Camera calibration
            {

                rapidjson::Value calibration = mat4x4_to_json_array(device->getDepthCamera()->getCalibration(), allocator);

                rapidjson::Value calibrationKey;
                calibrationKey.SetString("calibration");
                deviceJsonObj.AddMember(calibrationKey, calibration, allocator);
            }
            devices.PushBack(deviceJsonObj, allocator);
        }

        rapidjson::Value key;
        key.SetString("devices");
        d.AddMember(key, devices, allocator);
    }
    rapidjson::Value capture_time;
    auto time_str = currentTimeAsString();
    std::cout << "current time: " << time_str << std::endl;
    capture_time.SetString(rapidjson::StringRef(time_str));

    rapidjson::Value key;
    key.SetString("capture_time");
    d.AddMember(key, capture_time, allocator);

    std::ofstream out(filename);
    rapidjson::OStreamWrapper osw(out);

    rapidjson::PrettyWriter<rapidjson::OStreamWrapper> writer(osw);
    d.Accept(writer);
}