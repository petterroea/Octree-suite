#include "octreeSequence.h"
#include "encoding/octreeSequenceEncoder.h"

#include <iostream>

void printUsage() {
    std::cout << "octreeVideoEncoder - encodes octree video" << std::endl;
    std::cout << "Usage:" << std::endl;
    std::cout << "    octreeVideoEncoder [octreeFolder]" << std::endl;
}

int main(int argc, char** argv) {
    if(argc != 2) {
        printUsage();
        return 1;
    }

    OctreeSequence* sequence = new OctreeSequence(argv[1]);

    std::cout << "Loaded octree sequence with " << sequence->getFrameCount() << " frames @ " << sequence->getFps() << " fps." << std::endl;

    OctreeSequenceEncoder* encoder = new OctreeSequenceEncoder(sequence);
    encoder->encode();

    return 0;
}