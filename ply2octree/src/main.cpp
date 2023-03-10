#include <iostream>
#include <string>
#include <filesystem>

#include <getopt.h>

#include "octreeGenerator.h"
#include "pointcloud.h"

int main(int argc, char** argv) {
    std::cout << "ply2octree by petterroea(2022)" << std::endl;

    char* inputFilename = nullptr;
    char* outputFilename = nullptr;

    option longopts[] = {
        {"in", required_argument, NULL, 'i'},
        {"out", required_argument, NULL, 'o'}
    };

    while (true) {
        const int opt = getopt_long(argc, argv, "i:o:", longopts, 0);

        if (opt == -1) {
            break;
        }

        switch (opt) {
            case 'i':
                inputFilename = optarg;
                break;
            case 'o':
                outputFilename = optarg;
                break;
            default:
            break;
        }
    }
    if(inputFilename == nullptr) {
        std::cout << "Please specify an imput file" << std::endl;
        return 1;
    }
    if(outputFilename == nullptr) {
        std::cout << "Please specify an output file" << std::endl;
        return 1;
    }
    std::filesystem::path inputPath(inputFilename);
    std::filesystem::path outputPath(outputFilename);
    
    std::cout << "Loading " << inputFilename << " for converting to " << outputFilename << std::endl;
    if(!strcmp(inputFilename, outputFilename)) {
        std::cout << "Input and output are the same!" << std::endl;
        return 1;
    }

    if(!std::filesystem::exists(inputPath)) {
        std::cout << inputPath << " doesn't exist" << std::endl;
        return 1;
    }

    Pointcloud* file = nullptr;
    try {
        file = parsePlyFile(std::string(inputFilename));
    } catch(char* err) {
        std::cout << err << std::endl;
        return 1;
    }

    std::cout << "Converting..." << std::endl;

    OctreeGenerator generator(file);

    auto octree = generator.boxSortOuter(15);

    generator.writeToFile(octree, outputFilename);
}