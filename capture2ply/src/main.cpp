#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

#include <iostream>
#include <fstream>
#include <filesystem>

#include "capture.h"

void print_args() {
    std::cout << "capture2ply - convert depth camera captures to ply pointclouds" << std::endl;
    std::cout << "Usage:" << std::endl;
    std::cout << "\tcapture2ply [capture_directory] [output]" << std::endl;
}

int main(int argc, char** argv) {
    if(argc != 3) {
        print_args();
        return 1;
    }

    std::filesystem::path captureDirectory(argv[1]);
    std::filesystem::path captureMetadata("metadata.json");
    std::filesystem::path metadataPath = captureDirectory / captureMetadata;

    if(!std::filesystem::exists(metadataPath)) {
        std::cout << metadataPath << " does not exist - unable to load capture!" << std::endl;
        return 1;
    }

    std::cout << "Loading capture metadata from " << metadataPath << std::endl;

    std::ifstream ifs(metadataPath.string());
    rapidjson::IStreamWrapper isw(ifs);
    
    rapidjson::Document d;
    d.ParseStream(isw);
    Capture* capture = new Capture(d, captureDirectory);

    // Init output directory
    if(!std::filesystem::exists(argv[2])) {
        std::filesystem::create_directories(argv[2]);
    }
    capture->to_ply(argv[2]);
}