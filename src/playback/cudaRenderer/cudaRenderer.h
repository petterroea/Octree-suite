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

    cudaArray_t cuOutput = nullptr;
    cudaSurfaceObject_t outputSurfObj = 0;
    GLuint glOutput = 0;

    void mapGlToCu();

    // OpenGL rendering stuff
    CudaBlitShader shader;
    GLuint vao;
    GLuint vertexBuffer;
    GLuint texCoordBuffer;

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