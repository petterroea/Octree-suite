#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

#include <octree/octreeLoad.h>

#include "octreeSequence.h"

#include <exception>
#include <fstream>
#include <string>
#include <filesystem>

OctreeSequence::OctreeSequence(std::filesystem::path sequenceFolder) : sequenceFolder(sequenceFolder) {
    std::filesystem::path fileName("metadata.json");
    std::filesystem::path fullPath = sequenceFolder / fileName;

    std::ifstream ifs(fullPath.string());
    rapidjson::IStreamWrapper isw(ifs);

    rapidjson::Document d;
    d.ParseStream(isw);

    assert(d.IsObject());
    assert(d.HasMember("framecount"));
    this->frameCount = d["framecount"].GetInt();

    assert(d.HasMember("fps"));
    this->fps = d["fps"].GetFloat();
}

OctreeSequence::~OctreeSequence() {

}

PointerOctree<octreeColorType>* OctreeSequence::getOctree(int frame) {
    if(frame > this->frameCount || frame < 0) {
        throw std::invalid_argument("Invalid frame number");
    }
    std::filesystem::path fileName("capture_" + std::to_string(frame) + ".oct");
    std::filesystem::path fullPath = this->sequenceFolder / fileName;

    return loadOctree(fullPath.string());
}