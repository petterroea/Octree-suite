#include "octreeSequenceEncoder.h"
#include "encodingSequence.h"

#include <iostream>

OctreeSequenceEncoder::OctreeSequenceEncoder(OctreeSequence* sequence) : sequence(sequence) {

}

OctreeSequenceEncoder::~OctreeSequenceEncoder() {

}

void OctreeSequenceEncoder::encode() {
    int sequenceSize = 10;
    int currentAt = 0;
    while(currentAt < sequence->getFrameCount() - 1) {
        int length = std::min(
            this->sequence->getFrameCount() - currentAt - 1, 
            sequenceSize);
        std::cout << "Next length: " << length << std::endl;
        auto sequence = new EncodingSequence(
            this->sequence, 
            currentAt, 
            currentAt+length
        );
        sequence->encode();
        currentAt += length;
    }

}