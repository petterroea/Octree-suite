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

public:
    Capture(rapidjson::Document& document, std::filesystem::path workdir);
    ~Capture();

    void to_ply(std::string outputDirectory);
};