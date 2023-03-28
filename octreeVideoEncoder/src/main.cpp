#include "octreeSequence.h"
#include "encoding/octreeSequenceEncoder.h"

#include <iostream>
#include <filesystem>

void printUsage() {
    std::cout << "octreeVideoEncoder - encodes octree video" << std::endl;
    std::cout << "Usage:" << std::endl;
    std::cout << "    octreeVideoEncoder [octreeFolder] [outputFolder]" << std::endl;
}

int main(int argc, char** argv) {
    if(argc != 3) {
        printUsage();
        return 1;
    }

    OctreeSequence* sequence = new OctreeSequence(argv[1]);

    std::cout << "Loaded octree sequence with " << sequence->getFrameCount() << " frames @ " << sequence->getFps() << " fps." << std::endl;

    std::filesystem::path outputFolder(argv[2]);

    OctreeSequenceEncoder* encoder = new OctreeSequenceEncoder(sequence, outputFolder);
    encoder->encode();

    return 0;
}