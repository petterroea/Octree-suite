#pragma once

#include <rapidjson/document.h>

#include <glm/mat4x4.hpp>

#include <depthCamera/depthCamera.h>

#include <vector>
#include <string>
#include <filesystem>

class Capture {
    glm::mat4x4 captureTransform;
    std::vector<DepthCamera*> cameras;

    void writeMetadata(int framecount, float elapsedTime, std::filesystem::path& outputDir);

public:
    Capture(rapidjson::Document& document, std::filesystem::path& workdir);
    ~Capture();

    void to_ply(std::filesystem::path outputDir);
};