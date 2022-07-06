#include "octreeRenderer.h"

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <iostream>

#include "../gpuOctree.h"

#define THREADS_PER_BLOCK 64
#define BLOCKS 64

__device__ void swap(float& a, float& b) {
    float tmp = a;
    a = b;
    b = tmp;
}
// Assumes the position is always negative to the unit cube
__device__ void intersectFast(glm::vec3 cubeMin, glm::vec3 cubeMax, glm::vec3 p, glm::vec3 r, float& tcmin, float& tcmax) {
    //glm::vec3 cubeMaxTransformed(p.x < 1.0f ? cubeMin.x : cubeMax.x, p.y < 1.0f ? cubeMin.y : cubeMax.y, p.z < 1.0f ? cubeMin.z : cubeMax.z);
    float t0x = p.x > cubeMin.x ? 0.0f : ((cubeMin.x - p.x) / r.x);
    float t0y = p.y > cubeMin.y ? 0.0f : ((cubeMin.y - p.y) / r.y);
    float t0z = p.z > cubeMin.z ? 0.0f : ((cubeMin.z - p.z) / r.z);


    float t1x = (cubeMax.x - p.x) / r.x;
    float t1y = (cubeMax.y - p.y) / r.y;
    float t1z = (cubeMax.z - p.z) / r.z;
    tcmin = max(t0x, max(t0y, t0z));
    tcmax = min(t1x, min(t1y, t1z));
}
__device__ bool intersect(glm::vec3 cubeMin, glm::vec3 cubeMax, glm::vec3 p, glm::vec3 r, float& tmin_o, float& tmax_o) {
    float tmin = (cubeMin.x - p.x) / r.x; 
    float tmax = (cubeMax.x - p.x) / r.x; 
 
    if (tmin > tmax) swap(tmin, tmax); 
 
    float tymin = (cubeMin.y - p.y) / r.y; 
    float tymax = (cubeMax.y - p.y) / r.y; 
 
    if (tymin > tymax) swap(tymin, tymax); 
 
    if ((tmin > tymax) || (tymin > tmax)) 
        return false; 
 
    if (tymin > tmin) 
        tmin = tymin; 
 
    if (tymax < tmax) 
        tmax = tymax; 
 
    float tzmin = (cubeMin.z - p.z) / r.z; 
    float tzmax = (cubeMax.z - p.z) / r.z; 
 
    if (tzmin > tzmax) swap(tzmin, tzmax); 
 
    if ((tmin > tzmax) || (tzmin > tmax)) 
        return false; 
 
    if (tzmin > tmin) 
        tmin = tzmin; 
 
    if (tzmax < tmax) 
        tmax = tzmax; 
    
    tmin_o = tmin;
    tmax_o = tmax;
 
    return true; 
}

__global__ void kernel_cudaRender(GpuOctree* in, int rootOffset, cudaSurfaceObject_t output, int imgWidth, int imgHeight, glm::mat4x4* view, glm::mat4x4* projection, int flipFlag) {
    int pixelIdx = blockIdx.x*THREADS_PER_BLOCK+threadIdx.x;
    if(pixelIdx > imgWidth*imgHeight) {
        return;
    }
    int x = pixelIdx % imgWidth;
    int y = pixelIdx / imgWidth;

    float width_f = __int2float_rn(imgWidth);
    float height_f = __int2float_rn(imgHeight);
    
    float x_f = __int2float_rn(x);
    float y_f = __int2float_rn(y);

    //Ray stuff
    glm::vec4 p(0.0f, 0.0f, 0.0f, 1.0f);
    glm::vec4 r((x_f/width_f)*2.0f-1.0f, (1.0f - y_f/height_f)*2.0f-1.0f, -1.0f, 1.0f);

    // Move point from screen space to world space
    p = (*view) * p;
    // Make ray point in correct direction
    r = (*projection) * r;
    r.w = 0.0f;
    r = glm::normalize((*view) * r);

    // Store the sign of our ray directions
    int rayFlipFlagX = r.x<0.0f?0:1;
    int rayFlipFlagY = r.y<0.0f?0:1;
    int rayFlipFlagZ = r.z<0.0f?0:1;

    // XOR flags used to make sure ray is always increasing
    int flipFlagX = (flipFlag&1);
    int flipFlagY = (flipFlag>>1&1);
    int flipFlagZ = (flipFlag>>2&1);

    // Position is always negative
    p.x = copysignf(p.x, -1.0f);
    p.y = copysignf(p.y, -1.0f);
    p.z = copysignf(p.z, -1.0f);

    // Ray should be positive, but flip it based on what direction it used to point
    // as well as what quadrant we _actually_ are in
    r.x = copysignf(r.x, (rayFlipFlagX^flipFlagX)?1.0f:-1.0f);
    r.y = copysignf(r.y, (rayFlipFlagY^flipFlagY)?1.0f:-1.0f);
    r.z = copysignf(r.z, (rayFlipFlagZ^flipFlagZ)?1.0f:-1.0f);

    glm::vec3 cubeMin(-1.0f, -1.0f, -1.0f);
    glm::vec3 cubeMax(1.0f, 1.0f, 1.0f);
    glm::vec3 cubeCenter(0.0f, 0.0f, 0.0f);

    uint32_t color;

    float tmin_global = 0.0f;
    float tmax_global = 0.0f;

    // Intersect with the root node
    //Initialize raymarch variables
    float t = 0.0f;
    int currentNode = rootOffset;
    
    int scale = 0;
    int idx_x = 0;
    int idx_y = 0;
    int idx_z = 0;

    int raymarchStack[20];
    glm::vec3 centerStack[20];

        float tmin = 0.0f;
        float tmax = 0.0f;
    if(!intersect(cubeMin, cubeMax, glm::vec3(p.x, p.y, p.z), glm::vec3(r.x, r.y, r.z), tmin_global, tmax_global)) {
        color = 0;
        goto exitRaymarch;
    }
    t = tmin_global;

    //while(t < tmax_global) {

        intersectFast(cubeMin, cubeMax, glm::vec3(p.x, p.y, p.z), glm::vec3(r.x, r.y, r.z), tmin, tmax) ;
        {
            glm::vec3 colorVec = in[currentNode].color;

            glm::vec3 intersection = p+r*tmin;
            int intersectionX = flipFlagX^(intersection.x<0.0f?0:1);
            int intersectionY = flipFlagY^(intersection.y<0.0f?0:1);
            int intersectionZ = flipFlagZ^(intersection.z<0.0f?0:1);

            int r = __float2int_rd(colorVec.x*255.0f);
            int g = __float2int_rd(colorVec.y*255.0f);
            int b = __float2int_rd(colorVec.z*255.0f);
            color = (intersectionX ? ((r&0xff) << 0) : 0) | (intersectionY ? ((g&0xff) << 8) : 0) | (intersectionZ ? ((b&0xff) << 16) : 0);
            //color = (intersectionX ? (0xff << 0) : 0) | (intersectionY ? (0xff << 8) : 0) | (intersectionZ ? (0xff << 16) : 0);
            //TODO does this work with copysign?
            /*
            idx_x = idx_x << 1 | (intersection.x-cubeCenter.x)>0.0f?1:0;
            idx_y = idx_y << 1 | (intersection.y-cubeCenter.y)>0.0f?1:0;
            idx_z = idx_z << 1 | (intersection.z-cubeCenter.z)>0.0f?1:0;
            if(tmin > t) {
                // The intersection is in front of t, valid hit
                color = (idx_x)
            } else {
                // The intersection is behind t, invalid. Return to parent
            }
            */
        }
    //}

    /*    color = (intersection.x > 0 ? (0xff << 0) : 0) | (intersection.y > 0 ? (0xff << 8) : 0) | (intersection.z > 0 ? (0xff << 16) : 0);
    } else {
        color = 0;
    }*/

    
exitRaymarch:
    surf2Dwrite<uint32_t>(color, output, x*sizeof(uint32_t), y);
}

void cudaRender(void* in, int rootOffset, cudaSurfaceObject_t out, int imgWidth, int imgHeight, void* view, void* projection, int flipFlag) {
    std::cout << "Rendering on GPU..." << std::endl;
    CUDA_CATCH_ERROR
    //Calculate number of blocks we need to run
    int blockCount = (imgWidth*imgHeight)/THREADS_PER_BLOCK;
    kernel_cudaRender<<<blockCount, THREADS_PER_BLOCK>>>((GpuOctree*)in, rootOffset, out, imgWidth, imgHeight, (glm::mat4x4*)view, (glm::mat4x4*)projection, flipFlag);
    CUDA_CATCH_ERROR
    // Wait for compute to finish
    cudaDeviceSynchronize();
    CUDA_CATCH_ERROR
    std::cout << "Done rendering" << std::endl;
}
