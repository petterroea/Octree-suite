#include "octreeMetadata.h"

#include <fstream>
#include <iostream>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>


OctreeMetadata::OctreeMetadata(std::filesystem::path path) : path(path) {
    std::filesystem::path metadataPath = path / std::filesystem::path("metadata.json");
    if (!std::filesystem::exists(metadataPath)) {
        throw std::runtime_error("Metadata file does not exist: " + path.string());
    }

    std::ifstream ifs(metadataPath.string());
    rapidjson::IStreamWrapper isw(ifs);

    rapidjson::Document d;
    d.ParseStream(isw);

    if (d.HasParseError()) {
        throw std::runtime_error("Failed to parse metadata file: " + metadataPath.string());
    }

    assert(d.IsObject());

    auto frameSequence = d["frameSequence"].GetArray();

    for(auto& frame : frameSequence) {
        int start = frame["start"].GetInt();
        int end = frame["end"].GetInt();
        std::string filename = frame["filename"].GetString();

        std::filesystem::path fullPath = path / std::filesystem::path(filename);

        std::cout << "Frame " << start << " - " << end << " at " << fullPath.string() << std::endl;

        this->frames.push_back(new OctreeFrameset(start, end, fullPath.string()));
    }

    this->fps = d["fps"].GetFloat();
    this->frameCount = d["frameCount"].GetInt();
}

OctreeMetadata::~OctreeMetadata() {
    for(auto frame : this->frames) {
        delete frame;
    }
}

float OctreeMetadata::getFps() const {
    return this->fps;
}
int OctreeMetadata::getFrameCount() const {
    return this->frameCount;
}

OctreeFrameset* OctreeMetadata::getFramesetByFrame(int index) const {
    //TODO something smarter than seeking
    for(auto frame : this->frames) {
        if (frame->getStartIndex() <= index && frame->getEndIndex() >= index) {
            return frame;
        }
    }
    throw std::runtime_error("Frame not found");
    return nullptr;
}

std::filesystem::path OctreeMetadata::getPath() const {
    return this->path;
}