#pragma once
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>

#include <GL/glew.h>
#include <cuda_runtime.h>

#include "../../lib/octree.h"
#include "gpuOctree.h"

#include "../shaders/cudaBlitShader.h"

class CudaRenderer {
    int textureWidth;
    int textureHeight;

    // RGB image
    cudaArray_t cuOutputRgb = nullptr;
    cudaSurfaceObject_t outputSurfObjRgb = 0;
    GLuint glOutputRgb = 0;

    // Iteration count image
    cudaArray_t cuOutputIterations = nullptr;
    cudaSurfaceObject_t outputSurfObjIterations = 0;
    GLuint glOutputIterations= 0;

    // Texture lifecycle
    void cleanupTextures();
    
    void setupRgbTexture(int width, int height);
    void setupIterationTexture(int width, int height);

    void mapGlToCuda(GLuint glTexture, cudaArray_t* cudaArray, cudaSurfaceObject_t* surfaceObject);

    // OpenGL rendering stuff
    CudaBlitShader shader;
    GLuint vao;
    GLuint vertexBuffer;
    GLuint texCoordBuffer;

    int renderMode = 0;

    //Transformation matrices
    void* viewMatrixPtr;
    void* projectionMatrixPtr;

    // Convert octree structure to something GPU-friendly
    void* octreeGpuDataPtr;
    int rootNodeOffset = 0;
    void generateGpuOctree(Octree<glm::vec3>* octree);
public:
    CudaRenderer(Octree<glm::vec3>* octree);
    ~CudaRenderer();

    void updateTexture(int width, int height);
    void render(glm::mat4 view, glm::mat4 projection);
};