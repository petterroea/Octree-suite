#pragma once

#include <GL/glew.h>
#include <cuda_runtime.h>

#include <glm/mat4x4.hpp>

#include "../../../shaders/cudaBlitShader.h"
#include "../loader/octreeLoader.h"

class OctreeRenderer {
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

public:
    OctreeRenderer();
    ~OctreeRenderer();

    void updateTexture(int width, int height);
    void render(loadedOctreeType* currentFrameset, int frameIndex, glm::mat4 view, glm::mat4 projection);
    void setupGl();
};
