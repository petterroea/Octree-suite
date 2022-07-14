#include "cudaRenderer.h"

#include <iostream>
#include <vector>
#include <cstring>

#include "imgui.h"

#include <cuda_gl_interop.h>

#include <glm/mat4x4.hpp>

#include "kernels/octreeRenderer.h"

CudaRenderer::CudaRenderer(Octree<glm::vec3>* octree) {
    // Generate mesh
    glm::vec3 vertices[] = {
        glm::vec3(-1.0f, -1.0f, 0.0f),
        glm::vec3(1.0f, -1.0f, 0.0f),
        glm::vec3(1.0f, 1.0f, 0.0f),

        glm::vec3(-1.0f, -1.0f, 0.0f),
        glm::vec3(1.0f, 1.0f, 0.0f),
        glm::vec3(-1.0f, 1.0f, 0.0f),
    };


    glm::vec2 texCoords[] = {
        glm::vec2(0.0f, 1.0f),
        glm::vec2(1.0f, 1.0f),
        glm::vec2(1.0f, 0.0f),

        glm::vec2(0.0f, 1.0f),
        glm::vec2(1.0f, 0.0f),
        glm::vec2(0.0f, 0.0f),
    };

    std::cout << "Building buffers" << std::endl;
    // Build VAO, buffers, and upload the data
    //VAO
    glGenVertexArrays(1, &this->vao);
    glBindVertexArray(this->vao);

    //Buffers
    glGenBuffers(1, &this->vertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, this->vertexBuffer);
    glVertexAttribPointer(
        0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
        3,                  // size
        GL_FLOAT,           // type
        GL_FALSE,           // normalized?
        0,                  // stride
        (void*)0            // array buffer offset
    );
    glEnableVertexAttribArray(0);

    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3)*6, vertices, GL_STATIC_DRAW);

    glGenBuffers(1, &this->texCoordBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, this->texCoordBuffer);
    glVertexAttribPointer(
        1,                  
        2,                  // size
        GL_FLOAT,           // type
        GL_FALSE,           // normalized?
        0,                  // stride
        (void*)0            // array buffer offset
    );
    glEnableVertexAttribArray(1);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2)*6, texCoords, GL_STATIC_DRAW);

    //Cleanup
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    std::cout << "done building buffers, vao " << this->vao << " buffers " << this->vertexBuffer << " " << this->texCoordBuffer << std::endl;

    //Allocate transform matrixes
    cudaMalloc(&viewMatrixPtr, sizeof(glm::mat4x4));
    cudaMalloc(&projectionMatrixPtr, sizeof(glm::mat4x4));

    //Build data structure that fits the GPU
    this->generateGpuOctree(octree);
}

CudaRenderer::~CudaRenderer() {
    // Cleanup cuda side
    this->cleanupTextures();

    // Cleanup GL side
    glDeleteVertexArrays(1, &this->vao);
    glDeleteBuffers(1, &this->vertexBuffer);
    glDeleteBuffers(1, &this->texCoordBuffer);
}


void CudaRenderer::cleanupTextures() {
    std::cout << "Texture cleanup" << std::endl;
    // RGB
    glDeleteTextures(1, &this->glOutputRgb);
    cudaFreeArray(this->cuOutputRgb);
    cudaDestroySurfaceObject(this->outputSurfObjRgb);

    // Iteration count
    glDeleteTextures(1, &this->glOutputIterations);
    cudaFreeArray(this->cuOutputIterations);
    cudaDestroySurfaceObject(this->outputSurfObjIterations);
}

/*
 * Responsible for texture management
 * Both creating the initial texture and resizing it upon window size change
 */
void CudaRenderer::updateTexture(int width, int height) {
    if(width == this->textureWidth && height == this->textureHeight)
        return;
    std::cout << "Building new textures" << std::endl;

    this->textureWidth = width;
    this->textureHeight = height;
    // If we already have a texture, dispose of it first 
    if(this->glOutputRgb != 0) {
        this->cleanupTextures();
    }

    this->setupRgbTexture(width, height);
    this->setupIterationTexture(width, height);
}

void CudaRenderer::setupRgbTexture(int width, int height) {
    glGenTextures(1, &this->glOutputRgb);
    std::cout << "Allocated new RGB CUDA target texture " << this->glOutputRgb << std::endl;
    glBindTexture(GL_TEXTURE_2D, this->glOutputRgb);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); 

    // Upload pixels into texture
    void* placeholderData = malloc(width*height*3);
    for(int i = 0; i < width*height; i++) {
        unsigned char r = (i >> 16) & 0xff;
        unsigned char g = (i >> 8) & 0xff;
        unsigned char b = i & 0xff;
        ((unsigned char*)placeholderData)[i*3+0] = 0;
        ((unsigned char*)placeholderData)[i*3+1] = (i/width);
        ((unsigned char*)placeholderData)[i*3+2] = b;
    }
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, placeholderData);
    glBindTexture(GL_TEXTURE_2D, 0);
    free(placeholderData);

    this->mapGlToCuda(this->glOutputRgb, &this->cuOutputRgb, &this->outputSurfObjRgb);
}
void CudaRenderer::setupIterationTexture(int width, int height) {
    glGenTextures(1, &this->glOutputIterations);
    std::cout << "Allocated new RGB CUDA target texture " << this->glOutputIterations << std::endl;
    glBindTexture(GL_TEXTURE_2D, this->glOutputIterations);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); 

    // Upload pixels into texture
    void* placeholderData = malloc(width*height);
    for(int i = 0; i < width*height; i++) {
        ((unsigned char*)placeholderData)[i] = 0xFA;
    }
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, placeholderData);
    glBindTexture(GL_TEXTURE_2D, 0);
    free(placeholderData);

    this->mapGlToCuda(this->glOutputIterations, &this->cuOutputIterations, &this->outputSurfObjIterations);
}

void CudaRenderer::mapGlToCuda(GLuint glTexture, cudaArray_t* cudaArray, cudaSurfaceObject_t* surfaceObject) {
    // Create CUDA mapping
    cudaGraphicsResource_t graphicsResource;
    cudaGraphicsGLRegisterImage(&graphicsResource, glTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    cudaGraphicsMapResources(1, &graphicsResource, 0);
    cudaGraphicsSubResourceGetMappedArray(cudaArray, graphicsResource, 0, 0);
    cudaGraphicsUnmapResources(1, &graphicsResource, 0);

    // Create surface object(Used when reading/writing to the texture)
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;

    // Create the surface objects
    resDesc.res.array.array = *cudaArray;
    cudaCreateSurfaceObject(surfaceObject, &resDesc);
}

void CudaRenderer::render(glm::mat4 view, glm::mat4 projection) {
    // Upload matrixes
    glm::mat4x4 invertedView = glm::inverse(view);
    glm::mat4x4 invertedProj = glm::inverse(projection);
    cudaMemcpy(this->viewMatrixPtr, &invertedView, sizeof(glm::mat4x4), cudaMemcpyHostToDevice);
    cudaMemcpy(this->projectionMatrixPtr, &invertedProj, sizeof(glm::mat4x4), cudaMemcpyHostToDevice);

    // Calculate flip flags using center of the screen
    glm::vec4 p(0.0f, 0.0f, 0.0f, 1.0f);
    glm::vec4 r(0.0f, 0.0f, -1.0f, 1.0f);

    // Move point from screen space to world space
    p = (invertedView) * p;
    // Make ray point in correct direction
    r = (invertedProj) * r;
    r.w = 0.0f;
    r = glm::normalize((invertedView) * r);

    // XOR flags used to make sure ray is always increasing
    // TODO use copysign
    int flipFlag = (p.x<0.0f?0:1) | ((p.y<0.0f?0:1)<<1) | ((p.z<0.0f?0:1)<<2);

    ImGui::Text("Render mode");
    ImGui::RadioButton("RGB", &this->renderMode, 0); ImGui::SameLine();
    ImGui::RadioButton("Iteration count", &this->renderMode, 1); ImGui::SameLine();
    //Run CUDA
    cudaRender(this->octreeGpuDataPtr, this->rootNodeOffset, this->outputSurfObjRgb, this->outputSurfObjIterations, this->textureWidth, this->textureHeight, viewMatrixPtr, projectionMatrixPtr, flipFlag);

    //Blit the results
    glUseProgram(this->shader.getHandle());

    glUniform1i(this->shader.getTextureLocation(), 0);
    glActiveTexture(GL_TEXTURE0);
    switch(this->renderMode) {
        case 0:
            glBindTexture(GL_TEXTURE_2D, this->glOutputRgb);
            break;
        case 1:
            glBindTexture(GL_TEXTURE_2D, this->glOutputIterations);
            break;
        default:
            break;
    }

    glBindVertexArray(this->vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
}

int generateOctreeInternal(Octree<glm::vec3>* octree, std::vector<GpuOctree>& nodes) {
    GpuOctree gpuOctree(*octree->getPayload());
    for(int i = 0; i < 8; i++) {
        Octree<glm::vec3>* child = octree->getChildByIdx(i);
        if(child != nullptr) {
            gpuOctree.children[i] = generateOctreeInternal(child, nodes);
        } else {
            //TODO is 0-comparison faster?
            gpuOctree.children[i] = -1;
        }
    }
    nodes.push_back(gpuOctree);
    return nodes.size()-1;
}

void CudaRenderer::generateGpuOctree(Octree<glm::vec3>* octree) {
    std::vector<GpuOctree> nodes;

    generateOctreeInternal(octree, nodes);

    size_t octreeDataSize = sizeof(GpuOctree)*nodes.size();

    std::cout << "Octree GPU data size: " << (octreeDataSize / 1024) << "kb" << std::endl;
    cudaMalloc(&this->octreeGpuDataPtr, octreeDataSize);
    std::cout << "Allocated gpu octree data ptr: " << this->octreeGpuDataPtr << std::endl;
    cudaMemcpy(this->octreeGpuDataPtr, nodes.data(), octreeDataSize, cudaMemcpyHostToDevice);
    this->rootNodeOffset = nodes.size()-1;

    std::cout << "Uploaded to GPU" << std::endl;

}