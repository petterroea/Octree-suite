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
__device__ float intersectMinFast(glm::vec3 cubeMin, glm::vec3 p, glm::vec3 r) {
    float t0x = p.x > cubeMin.x ? -100000.0f : ((cubeMin.x - p.x) / r.x);
    float t0y = p.y > cubeMin.y ? -100000.0f : ((cubeMin.y - p.y) / r.y);
    float t0z = p.z > cubeMin.z ? -100000.0f : ((cubeMin.z - p.z) / r.z);

    return max(t0x, max(t0y, t0z));
}

// Position is never positive, so we don't have the same issue with the max intersection
// Does not handle situations where the camera is between the min and max coordinate for a given axis
__device__ float intersectMaxFast(glm::vec3 cubeMax, glm::vec3 p, glm::vec3 r) {
    float t1x = ((cubeMax.x - p.x) / r.x);
    float t1y = ((cubeMax.y - p.y) / r.y);
    float t1z = ((cubeMax.z - p.z) / r.z);
    return min(t1x, min(t1y, t1z));
}

/* Assumes position is always negative
 * intersectMaxFast requires t0x, t0y, and t0z in order to be correct.
 * This is because both sides for a given acis may be "max" if the camera is sliced between them
 * 
 * t0x, t0y, and t0z also have to be adjusted such that they are not considered if the camera is 
 * "behind" the surfaces of the planes defined by cubeMin, AFTER the values are used by t1x, t1y, t1z calculations.
 */
// NOTE: not actually fast
__device__ bool intersectFast(glm::vec3 cubeMin, glm::vec3 cubeMax, glm::vec3 p, glm::vec3 r, float& tmin_o, float& tmax_o) {
    float t0x = ((cubeMin.x - p.x) / r.x);
    float t1x = ((cubeMax.x - p.x) / r.x);
    if(p.x > cubeMin.x) {
        t1x = max(t1x, t0x);
        t0x = 0.0f;
    }

    float t0y = ((cubeMin.y - p.y) / r.y);
    float t1y = ((cubeMax.y - p.y) / r.y);
    if(p.y > cubeMin.y) {
        t1y = max(t1y, t0y);
        t0y = 0.0f;
    }

    float t0z = ((cubeMin.z - p.z) / r.z);
    float t1z = ((cubeMax.z - p.z) / r.z);
    if(p.z > cubeMin.z) {
        t1z = max(t1z, t0z);
        t0z = 0.0f;
    }
/*
    float t1x = ((cubeMax.x - p.x) / r.x);
    t1x = p.x > cubeMin.x ? max(t1x, t0x) : t1x;
    float t1y = ((cubeMax.y - p.y) / r.y);
    t1y = p.y > cubeMin.y ? max(t1y, t0y) : t1y;
    float t1z = ((cubeMax.z - p.z) / r.z);
    t1z = p.z > cubeMin.z ? max(t1z, t0z) : t1z;

    t0x = p.x > cubeMin.x ? 0.0f : t0x;
    t0y = p.y > cubeMin.y ? 0.0f : t0y;
    t0z = p.z > cubeMin.z ? 0.0f : t0z;
*/
    tmin_o = max(t0x, max(t0y, t0z));

    tmax_o = min(t1x, min(t1y, t1z));

    return tmin_o <= tmax_o;
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

__global__ void kernel_cudaRender(GpuOctree* in, int rootOffset, cudaSurfaceObject_t output, cudaSurfaceObject_t iterationOutput, int imgWidth, int imgHeight, glm::mat4x4* view, glm::mat4x4* projection, int flipFlag) {
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

    uint32_t color = 0xFFCCCC;

    float tmin_global = 0.0f;
    float tmax_global = 0.0f;

    // Intersect with the root node
    //Initialize raymarch variables
    float t = 0.0f;
    
    int scale = 0;
    int idx_x = 0;
    int idx_y = 0;
    int idx_z = 0;

    int raymarchStack[20];
    glm::vec3 centerStack[20];
    
    // Init the top stack entry
    raymarchStack[0] = rootOffset;
    centerStack[0] = glm::vec3(0.0f, 0.0f, 0.0f);

    float tmin = 0.0f;
    float tmax = 0.0f;

    int intersected = 0;

    // If we don't hit the root node, immediately disregard the ray
    bool inCube = intersect(cubeMin, cubeMax, glm::vec3(p.x, p.y, p.z), glm::vec3(r.x, r.y, r.z), tmin_global, tmax_global);
    if(!inCube) {
        // Do nothing
    }  else {
        t = tmin_global;
        int iterations = 0;
        while(t < tmax_global && iterations < 100) {
            iterations++;
            if(scale * 20> color) {
                //color = scale * 20;
            }
            //Find a child to iterate to
            float childScale = powf(2.0f, -(scale+1));

            int validIndices[8];
            int validCount = 0;
            // Find the closest cube that intersects
            float lowestT = 1000.0f;
            int lowestTIdx = 0;

            bool isLeaf = true;
            for(int i = 0; i < 8; i++) {
                // Is there a child there?
                if(in[raymarchStack[scale]].children[i^flipFlag] != -1) {
                    isLeaf = false;
                    glm::vec3 childOffset = centerStack[scale] + glm::vec3(
                        copysign(childScale, ((i) & 0x1) == 1 ?-1.0f:1.0f),
                        copysign(childScale, ((i) & 0x2) == 2 ?-1.0f:1.0f),
                        copysign(childScale, ((i) & 0x4) == 4 ?-1.0f:1.0f) 
                    );
                    glm::vec3 childMin = cubeMin*childScale + childOffset;
                    glm::vec3 childMax = cubeMax*childScale + childOffset;
                    // Do we intersect with the child?
                    if(intersect(childMin, childMax, glm::vec3(p.x, p.y, p.z), glm::vec3(r.x, r.y, r.z), tmin, tmax)) {
                        // Is the intersection in front of the current ray progress?
                        if(tmin >= t) {
                            validIndices[validCount++] = i;
                            if(tmin < lowestT) {
                                lowestT = tmin;
                                lowestTIdx = i;
                            }
                        }
                    }
                }
            }
            // No children = leaf node. Return color and exit
            if(isLeaf) {
                glm::vec3 colorVec = in[raymarchStack[scale]].color;

                float myScale = powf(2.0f, -(scale));
                /*
                glm::vec3 intersection = (p+r*tmin);
                intersection = (intersection - centerStack[scale])*myScale;
                int intersectionX = flipFlagX^(intersection.x<0.0f?0:1);
                int intersectionY = flipFlagY^(intersection.y<0.0f?0:1);
                int intersectionZ = flipFlagZ^(intersection.z<0.0f?0:1);
*/
                int r = __float2int_rd(colorVec.x*255.0f);
                int g = __float2int_rd(colorVec.y*255.0f);
                int b = __float2int_rd(colorVec.z*255.0f);

                //color = (intersectionX ? ((r&0xff) << 0) : 0) | (intersectionY ? ((g&0xff) << 8) : 0) | (intersectionZ ? ((b&0xff) << 16) : 0);
                color = ((r&0xff) << 0) | ((g&0xff) << 8) | ((b&0xff) << 16);
                break;
            }
            // We have a child to evaluate, push it to the stack
            if(validCount > 0) {
                //color = 0xFF0000 | (scale << 8) | lowestTIdx;
                int nextIndex = lowestTIdx;
                int nextNode = in[raymarchStack[scale]].children[nextIndex^flipFlag];
                scale++;
                centerStack[scale] = centerStack[scale-1] + glm::vec3(
                    copysign(childScale, (nextIndex & 0x1) == 1 ?-1.0f:1.0f),
                    copysign(childScale, (nextIndex & 0x2) == 2 ?-1.0f:1.0f),
                    copysign(childScale, (nextIndex & 0x4) == 4 ?-1.0f:1.0f) 
                );
                raymarchStack[scale] = nextNode;
                //t = lowestT;
            } else {
                // We did not hit anything, return to parent(with updated t)
                if(scale==0) {
                    //color = 0x00FF00;
                    break;
                }

                float myScale = powf(2.0f, -(scale));

                // Settle for this voxel if we are returning from a high scale
                if(scale==9) {
                    glm::vec3 colorVec = in[raymarchStack[scale]].color;

                    int r = __float2int_rd(colorVec.x*255.0f);
                    int g = __float2int_rd(colorVec.y*255.0f);
                    int b = __float2int_rd(colorVec.z*255.0f);

                    //color = (intersectionX ? ((r&0xff) << 0) : 0) | (intersectionY ? ((g&0xff) << 8) : 0) | (intersectionZ ? ((b&0xff) << 16) : 0);
                    color = ((r&0xff) << 0) | ((g&0xff) << 8) | ((b&0xff) << 16);
                    break;
                }

                //No hits, proceed forward
                // Set t to tmax of the current node
                glm::vec3 myMin = cubeMin*myScale + centerStack[scale];
                glm::vec3 myMax = cubeMax*myScale + centerStack[scale];
                intersect(myMin, myMax, glm::vec3(p.x, p.y, p.z), glm::vec3(r.x, r.y, r.z), tmin, tmax);
                t = tmax;
                scale--;
            }
            //tmin = intersectMinFast(cubeMin, glm::vec3(p.x, p.y, p.z), glm::vec3(r.x, r.y, r.z));
        }
        surf2Dwrite<uint8_t>(iterations*8, iterationOutput, x*sizeof(uint8_t), y);
    }

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
    //}

    /*    color = (intersection.x > 0 ? (0xff << 0) : 0) | (intersection.y > 0 ? (0xff << 8) : 0) | (intersection.z > 0 ? (0xff << 16) : 0);
    } else {
        color = 0;
    }*/
    
exitRaymarch:
    surf2Dwrite<uint32_t>(color, output, x*sizeof(uint32_t), y);
}

void cudaRender(void* in, int rootOffset, cudaSurfaceObject_t out, cudaSurfaceObject_t iterations, int imgWidth, int imgHeight, void* view, void* projection, int flipFlag) {
    std::cout << "Rendering on GPU..." << std::endl;
    CUDA_CATCH_ERROR
    //Calculate number of blocks we need to run
    int blockCount = (imgWidth*imgHeight)/THREADS_PER_BLOCK;
    kernel_cudaRender<<<blockCount, THREADS_PER_BLOCK>>>((GpuOctree*)in, rootOffset, out, iterations, imgWidth, imgHeight, (glm::mat4x4*)view, (glm::mat4x4*)projection, flipFlag);
    CUDA_CATCH_ERROR
    // Wait for compute to finish
    cudaDeviceSynchronize();
    CUDA_CATCH_ERROR
    std::cout << "Done rendering" << std::endl;
}
