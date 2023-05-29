#pragma once
#include "../config.h"

#include <yuv.h>

template <typename T>
class OctreeProcessingPayload {
public:
    T data;
    glm::vec3 yuv;
    int replacement;
    int writtenOffset;
    bool trimmed;

    OctreeProcessingPayload(T data);
    OctreeProcessingPayload(const OctreeProcessingPayload<T>& old);
};

template <typename T>
OctreeProcessingPayload<T>::OctreeProcessingPayload(const OctreeProcessingPayload<T>& old) {
    this->data = old.data;
    this->replacement = -1;
    this->writtenOffset = -1;
    this->trimmed = false;
    this->yuv = old.yuv;
}