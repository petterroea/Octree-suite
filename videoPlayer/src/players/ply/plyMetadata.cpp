#include "plyMetadata.h"

#include <fstream>
#include <iostream>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

PlyMetadata::PlyMetadata(std::filesystem::path path): path(path) {
    std::filesystem::path metadataPath = path / std::filesystem::path("metadata.json");
    if (!std::filesystem::exists(metadataPath)) {
        throw std::runtime_error("Metadata file does not exist: " + path.string());
    }
    std::cout << "Reading metadata from " << metadataPath.string() << std::endl;

    std::ifstream ifs(metadataPath.string());
    rapidjson::IStreamWrapper isw(ifs);

    rapidjson::Document d;
    d.ParseStream(isw);

    if (d.HasParseError()) {
        throw std::runtime_error("Failed to parse metadata file: " + metadataPath.string());
    }
    std::cout << "Successfully parsed metadata" << std::endl;

    assert(d.IsObject());

    this->fps = d["fps"].GetFloat();
    this->frameCount = d["framecount"].GetInt();

    std::cout << "Successfully read metadata" << std::endl;
}

float PlyMetadata::getFps() const {
    return this->fps;
}
int PlyMetadata::getFrameCount() const {
    return this->frameCount;
}
std::filesystem::path PlyMetadata::getPath() const { 
    return this->path;
}