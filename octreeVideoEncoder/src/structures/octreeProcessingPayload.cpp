#include "octreeProcessingPayload.h"

#include <iostream>

template <>
OctreeProcessingPayload<glm::vec3>::OctreeProcessingPayload(glm::vec3 data): data(data){
    this->replacement = -1;
    this->writtenOffset = -1;
    this->trimmed = false;
    this->yuv = rgb_to_yuv(data);
    /*
    std::cout << "Converted RGB to YUV:" << std::endl;
    std::cout << data.x << " " << data.y << " " << data.z << std::endl;
    std::cout << yuv.x << " " << yuv.y << " " << yuv.z << std::endl;
    */
}