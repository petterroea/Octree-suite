#pragma once

#include <filesystem>
#include <vector>

class PlyMetadata {
    float fps;
    int frameCount;

    std::filesystem::path path;
public:
    PlyMetadata(std::filesystem::path path);
    
    float getFps() const;
    int getFrameCount() const;
    std::filesystem::path getPath() const;
};