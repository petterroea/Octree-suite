#include "cudaTransformPointCloud.h"

#define THREADS_PER_BLOCK 32

// Applies a matrix transform to each vector in an array of vertices
__global__ void kernel_transformPointCloud(
    // Inputs
    glm::vec3* pointSrc, 
    cudaTextureObject_t colorSrc, 
    glm::vec2* texCoordSrc,
    // Outputs
    glm::vec3* pointDst, 
    glm::vec3* colorDst, 
    // Params
    int count, 
    glm::mat4x4 transform) {
    int idx = blockIdx.x*THREADS_PER_BLOCK+threadIdx.x;
    if(idx >= count) {
        return;
    }
    glm::vec3 vec = pointSrc[idx];
    glm::vec4 tmp(vec.x, vec.y, vec.z, 1.0f);
    tmp = transform * tmp;
    pointDst[idx].x = tmp.x < -1.0f ? 0.0f : (tmp.x > 1.0f ? 0.0f : tmp.x);
    pointDst[idx].y = tmp.y < -1.0f ? 0.0f : (tmp.y > 1.0f ? 0.0f : tmp.y);
    pointDst[idx].z = tmp.z < -1.0f ? 0.0f : (tmp.z > 1.0f ? 0.0f : tmp.z);
    // Sample the color
    float4 color;
    tex2D<float4>(&color, colorSrc, texCoordSrc[idx].x, texCoordSrc[idx].y);

    colorDst[idx].x = color.x;
    colorDst[idx].y = color.y;
    colorDst[idx].z = color.z;
}

void cudaTransformPointCloud(
    glm::vec3* pointSrc, 
    cudaTextureObject_t colorSrc, 
    glm::vec2* texCoordSrc, 
    glm::vec3* pointDst, 
    glm::vec3* colorDst, 
    int count, 
    glm::mat4x4 transform) {
    int blockCount = (count+1)/THREADS_PER_BLOCK;
    kernel_transformPointCloud<<<blockCount, THREADS_PER_BLOCK>>>(pointSrc, colorSrc, texCoordSrc, pointDst, colorDst, count, transform);
}