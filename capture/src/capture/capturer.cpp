#include "capturer.h"

#include <utils.h>

#include <imgui.h>

#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

#include <rapidjson/document.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/writer.h>

#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <fstream>
#include <filesystem>


namespace fs = std::filesystem;

Capturer::Capturer(std::vector<DepthCamera*> cameras) : cameras(cameras), writer(1920*1080*cameras.size()) {

}

void Capturer::getFrame() {
    std::cout << "- getFrame called" << std::endl;
    for(auto device : this->cameras) {
        if(this->autoCalibrate) {
            device->setCalibrationEnabled(true);
        }
        device->requestFrame();
    }
    for(auto device : this->cameras) {
        device->waitForNewFrame();
    }
    if(this->autoCalibrate) {
        // Check if all cameras see enough markers
        bool allGood = true;
        for(auto device : this->cameras) {
            if(!device->getCalibrator().getIsValidPose() || device->getCalibrator().getLastDetectedMarkers() < this->autoCalibrateTreshold) {
                allGood = false;
            }
        }
        if(allGood) {
            this->autoCalibrate = false;

            for(auto device : this->cameras) {
                device->setCalibrationEnabled(false);
            }
        }
    }
    //Optional
    for(auto device : this->cameras) {
        device->uploadGpuDataSync();
    }
}

void Capturer::saveCalibration(std::string filename) {
    rapidjson::Document d;
    d.SetObject();
    rapidjson::Document::AllocatorType& allocator = d.GetAllocator();
    for(auto device : this->cameras) {
        glm::mat4x4 matrix = device->getCalibration();
        rapidjson::Value deviceMatrixArray;
        deviceMatrixArray.SetArray();
        deviceMatrixArray
            .PushBack(matrix[0][0], allocator).PushBack(matrix[0][1], allocator).PushBack(matrix[0][2], allocator).PushBack(matrix[0][3], allocator)
            .PushBack(matrix[1][0], allocator).PushBack(matrix[1][1], allocator).PushBack(matrix[1][2], allocator).PushBack(matrix[1][3], allocator)
            .PushBack(matrix[2][0], allocator).PushBack(matrix[2][1], allocator).PushBack(matrix[2][2], allocator).PushBack(matrix[2][3], allocator)
            .PushBack(matrix[3][0], allocator).PushBack(matrix[3][1], allocator).PushBack(matrix[3][2], allocator).PushBack(matrix[3][3], allocator);

        rapidjson::Value key(device->getSerial().c_str(), allocator); // copy string name
        d.AddMember(key, deviceMatrixArray, allocator);
    }

    std::ofstream out(filename);
    rapidjson::OStreamWrapper osw(out);

    rapidjson::Writer<rapidjson::OStreamWrapper> writer(osw);
    d.Accept(writer);
}

void Capturer::loadCalibration() {
    std::ifstream ifs("calibration.json");
    rapidjson::IStreamWrapper isw(ifs);
    
    rapidjson::Document d;
    d.ParseStream(isw);

    bool error = false;
    bool incomplete = false;

    for(auto device : this->cameras) {
        rapidjson::Value::ConstMemberIterator itr = d.FindMember(device->getSerial().c_str());
        if (itr != d.MemberEnd()) {
            rapidjson::Value& a = d[device->getSerial().c_str()];
            if(a.Capacity() < 4*4) {
                error = true;
                break;
            }
            glm::mat4x4 calibration(1.0f); 
            calibration[0][0] = a[0].GetFloat(); calibration[0][1] = a[1].GetFloat(); calibration[0][2] = a[2].GetFloat(); calibration[0][3] = a[3].GetFloat();
            calibration[1][0] = a[4].GetFloat(); calibration[1][1] = a[5].GetFloat(); calibration[1][2] = a[6].GetFloat(); calibration[1][3] = a[7].GetFloat();
            calibration[2][0] = a[8].GetFloat(); calibration[2][1] = a[9].GetFloat(); calibration[2][2] = a[10].GetFloat(); calibration[2][3] = a[11].GetFloat();
            calibration[3][0] = a[12].GetFloat(); calibration[3][1] = a[13].GetFloat(); calibration[3][2] = a[14].GetFloat(); calibration[3][3] = a[15].GetFloat();
            device->setCalibration(calibration);
        } else {
            incomplete = true;
        }
    }
}

void Capturer::render(glm::mat4x4& view, glm::mat4x4& proj) {
    for(auto device : this->cameras) {
        device->getRenderer()->render(device->getCalibration(), view, proj, device->getPointCount());
    }
    this->settings.renderHelpLines(view, proj);

    if(this->videoCapture) {
        this->framesCaptured++;
    }
}

void Capturer::capture() {
    std::vector<Pointcloud> pointclouds;
    // If the thread isn't done writing the last thread(unlikely), wait
    this->writer.waitForSafeToWrite();
    // Calculate the transformation matrix for the configured capture region
    glm::mat4x4 captureTransform = glm::scale(
        glm::translate(
            glm::mat4(1.0f), 
            -1.0f*this->settings.getCapturePosition()), 
        glm::vec3(1.0f/this->settings.getCaptureScale())
    );
    // Transform points to world space and download them from the GPU
    for(auto device : this->cameras) {
        Pointcloud pointcloud;
        device->capturePoints(&pointcloud.points, &pointcloud.colors, &pointcloud.count, captureTransform);
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
            for(auto camera : this->cameras) {
                camera->endCaptureThread();
            }
            std::cout << "Sent stop signal to threads" << std::endl;
            for(auto camera : this->cameras) {
                camera->waitForThreadJoin();
            }
            std::cout << "Threads joined" << std::endl;
            for(auto camera : this->cameras) {
                camera->beginStreaming();
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
            this->saveCalibration(capture_directory + "/camera_poses.json");
            this->saveVideoMetadata(capture_directory + "/metadata.json");

            this->videoCapture = true;
            this->captureStart = std::chrono::system_clock::now();
            for(auto camera : this->cameras) {
                camera->endCaptureThread();
            }
            for(auto camera : this->cameras) {
                camera->waitForThreadJoin();
            }
            std::cout << "Threads joined" << std::endl;
            for(auto camera : this->cameras) {
                camera->beginRecording(capture_directory + "/capture-" + camera->getSerial() + ".bag");
            }
        }
    }
    ImGui::End();
    ImGui::Begin("Devices");
    for(auto device : this->cameras) {
        ImGui::PushID(device->getSerial().c_str());

        char label[200];
        sprintf(label, "%s: %s", device->getKind().c_str(), device->getSerial().c_str());
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
        version.SetString("realsense");

        rapidjson::Value key;
        key.SetString("make");
        d.AddMember(key, version, allocator);
    }
    {
        rapidjson::Value version;
        version.SetString("1");

        rapidjson::Value key;
        key.SetString("version");
        d.AddMember(key, version, allocator);
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

    rapidjson::Writer<rapidjson::OStreamWrapper> writer(osw);
    d.Accept(writer);
}