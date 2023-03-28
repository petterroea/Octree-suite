#include "octreeSequenceEncoder.h"
#include "encodingSequence.h"

#include <rapidjson/document.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/prettywriter.h>

#include <iostream>
#include <fstream>
#include <string>

OctreeSequenceEncoder::OctreeSequenceEncoder(OctreeSequence* sequence, std::filesystem::path outputFolder) : sequence(sequence), outputFolder(outputFolder) {
    // Make sure we have an output dir
    if(!std::filesystem::exists(this->outputFolder)) {
        std::filesystem::create_directories(this->outputFolder);
    } else if (!std::filesystem::is_directory(this->outputFolder)) {
        throw std::runtime_error("Output folder is not a directory");
    } 
}

OctreeSequenceEncoder::~OctreeSequenceEncoder() {

}

void OctreeSequenceEncoder::encode() {
    std::filesystem::path metadataFile = outputFolder / "metadata.json";
    if(std::filesystem::exists(metadataFile)) {
        throw std::runtime_error("Metadata file already exists");
    }

    rapidjson::Document d;
    d.SetObject();
    rapidjson::Document::AllocatorType& allocator = d.GetAllocator();

    rapidjson::Value frameSequence(rapidjson::kArrayType);

    int sequenceSize = 5;
    int currentAt = 0;
    while(currentAt < sequence->getFrameCount() - 1) {
        int length = std::min(
            this->sequence->getFrameCount() - currentAt - 1, 
            sequenceSize);
        std::cout << "Next length: " << length << std::endl;

        auto from = currentAt;
        auto to = currentAt + length;

        std::string filename = std::to_string(from) + "-" + std::to_string(to) + std::string(".loc");

        std::filesystem::path fullSequencePath = outputFolder / filename;

        auto sequence = new EncodingSequence(
            this->sequence, 
            from, 
            to,
            fullSequencePath.string()
        );
        sequence->encode();
        currentAt += length;

        rapidjson::Value frameFilename;
        frameFilename.SetString(filename.c_str(), allocator);
        frameSequence.PushBack(frameFilename, allocator);
    }

    d.AddMember("frameSequence", frameSequence, allocator);
    std::ofstream out(metadataFile.string());
    rapidjson::OStreamWrapper osw(out);
    rapidjson::PrettyWriter<rapidjson::OStreamWrapper> writer(osw);
    d.Accept(writer);

    std::cout << "Done writing metadata file, thank you come again!" << std::endl;
}