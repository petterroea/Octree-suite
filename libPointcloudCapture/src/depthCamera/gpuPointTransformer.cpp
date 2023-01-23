#include "gpuPointTransformer.h"

#include <iostream>

#include <cudaHelpers.h>

#include "../kernels/cudaTransformPointCloud.h"

GpuPointTransformer::GpuPointTransformer(VideoMode mode) {
    // Allocate buffers for transforming buffers
    int expectedMaxPointCount = mode.colorWidth * mode.colorHeight;
    std::cout << "Expected max points: " << expectedMaxPointCount << std::endl;

    cudaMalloc(&this->devPtrPointsTransformed, expectedMaxPointCount * sizeof(glm::vec3));
    cudaMalloc(&this->devPtrPointColors, expectedMaxPointCount * sizeof(glm::vec3));
    for(int i = 0; i < 2; i++) {
        hostPointsTransformed[i] = new glm::vec3[expectedMaxPointCount];
        hostColors[i] = new glm::vec3[expectedMaxPointCount];
    }
}
GpuPointTransformer::~GpuPointTransformer() {
    cudaFree(this->devPtrPointsTransformed);
    cudaFree(this->devPtrPointColors);

    for(int i = 0; i < 2; i++) {
        delete[] hostPointsTransformed[i];
        delete[] hostColors[i];
    }
}

void GpuPointTransformer::transformPoints(void* pointBuffer, cudaTextureObject_t texture, void* textureCoords, int pointCount, glm::mat4x4 transform) {
    std::cout << "Transforming points to " << devPtrPointsTransformed << std::endl;
    CUDA_CATCH_ERROR
    cudaTransformPointCloud(
        (glm::vec3*)pointBuffer, 
        texture, 
        (glm::vec2*)textureCoords,
        (glm::vec3*)this->devPtrPointsTransformed,
        (glm::vec3*)this->devPtrPointColors,
        pointCount, 
        transform 
    );
    cudaDeviceSynchronize();
    CUDA_CATCH_ERROR
    cudaMemcpy(this->hostPointsTransformed[currentBuffer], this->devPtrPointsTransformed, pointCount*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->hostColors[currentBuffer], this->devPtrPointColors, pointCount*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
}

void GpuPointTransformer::getBuffers(glm::vec3** points, glm::vec3** colors) {
    *points = this->hostPointsTransformed[currentBuffer];
    *colors = this->hostColors[currentBuffer];
}
void GpuPointTransformer::swapBuffers() {
    currentBuffer = (currentBuffer + 1) % 2;
}