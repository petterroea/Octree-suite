#include "depthCamera.h"

#include <imgui.h>

#include <cuda_gl_interop.h>

#include <librealsense2/hpp/rs_device.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include <iostream>

#include <cudaHelpers.h>

#include "openCVCalibrator.h"
#include "../render/shaders/pointcloudShader.h"

DepthCamera::DepthCamera(RenderMode renderMode, VideoMode videoMode) : calibratedTransform(1.0f), renderMode(renderMode), videoMode(videoMode) {
    if(renderMode == RenderMode::HEADLESS) {
        this->setupGpuMemoryHeadless(videoMode);
    } else {
        this->setupGpuMemoryOpenGL(videoMode);
    }
}

PointcloudShader* shaderSingleton = nullptr;

void DepthCamera::setupGpuMemoryOpenGL(VideoMode mode) {
    if(shaderSingleton == nullptr) {
        shaderSingleton = new PointcloudShader();
    }
    // Let the renderer set up OpenGL handles, then bind them to CUDA handles
    this->renderer = new TexturedPointcloudRenderer(videoMode, shaderSingleton);
    // Map textures to CUDA
    this->mapGlTextureToCuda(this->renderer->getColorTextureHandle(), &this->cuArrayTexRgb, &this->cuSurfaceObjectTexRgb);
    this->mapGlTextureToCuda(this->renderer->getDepthTextureHandle(), &this->cuArrayTexDepth, &this->cuSurfaceObjectTexDepth);

    // Map buffers to CUDA
    this->mapGlBufferToCuda(this->renderer->getPointBufferHandle(), &this->devPtrPoints);
    this->mapGlBufferToCuda(this->renderer->getTextureCoordBufferHandle(), &this->devPtrTexCoords);
}


void DepthCamera::mapGlTextureToCuda(GLuint glTexture, cudaArray_t* cudaArray, cudaSurfaceObject_t* surfaceObject) {
    // Create CUDA mapping
    cudaGraphicsResource_t graphicsResource;
    cudaGraphicsGLRegisterImage(&graphicsResource, glTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    CUDA_CATCH_ERROR
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

void DepthCamera::mapGlBufferToCuda(GLuint glBuffer, void** devPtr) {
    // Create CUDA mapping
    cudaGraphicsResource_t graphicsResource;
    //TODO different usage will require different flags
    cudaGraphicsGLRegisterBuffer(&graphicsResource, glBuffer, cudaGraphicsRegisterFlagsNone);
    CUDA_CATCH_ERROR
    cudaGraphicsMapResources(1, &graphicsResource, 0);
    CUDA_CATCH_ERROR
    cudaGraphicsResourceGetMappedPointer(devPtr, nullptr, graphicsResource);
    CUDA_CATCH_ERROR
    cudaGraphicsUnmapResources(1, &graphicsResource, 0);
}

void DepthCamera::setupGpuMemoryHeadless(VideoMode mode) {
    throw "TODO";
}

DepthCamera::~DepthCamera() {
    if(this->renderer != nullptr) {
        delete this->renderer;
    }
    //TODO unregister
    cudaFreeArray(this->cuArrayTexRgb);
    cudaDestroySurfaceObject(this->cuSurfaceObjectTexRgb);

    cudaFreeArray(this->cuArrayTexDepth);
    cudaDestroySurfaceObject(this->cuSurfaceObjectTexDepth);
/*
    cudaFreeArray(this->cuArrayPoints);
    cudaDestroySurfaceObject(this->cuSurfaceObjectPoints);

    cudaFreeArray(this->cuArrayTexCoords);
    cudaDestroySurfaceObject(this->cuSurfaceObjectTexCoords);
    */
}

void DepthCamera::drawImmediateGui() {
    ImGui::Checkbox("Enable OpenCV", &this->calibrationEnabled);
    this->calibrator.drawImmediateGui();
}

void DepthCamera::startCaptureThread() {
    sem_init(&this->frameRequestSemaphore, 0, 0);
    sem_init(&this->frameReceivedSemaphore, 0, 0);
    this->hThread = pthread_create(&this->hThread, NULL, (void* (*)(void*))DepthCamera::threadEntrypoint, (void*)this);
}

void DepthCamera::endCaptureThread() {
    this->running = false;
    // Make sure the loop gets to finish
    this->requestFrame();
}

void DepthCamera::waitForThreadJoin() {
    pthread_join(this->hThread, NULL);
}

void DepthCamera::threadEntrypoint(DepthCamera* self) {
    // processingThread is a virtual function implemented by the device "driver"
    self->processingThread();
}

void DepthCamera::requestFrame() {
    std::cout << this->getSerial() << " Requesting frame" << std::endl;
    sem_post(&this->frameRequestSemaphore);
}
void DepthCamera::waitForNewFrame() {
    sem_wait(&this->frameReceivedSemaphore);
}

void DepthCamera::processingThread() {
    this->beginCapture();

    while(this->running) {
        sem_wait(&this->frameRequestSemaphore);
        this->processFrame();

        sem_post(&this->frameReceivedSemaphore);
    }
    std::cout << this->getSerial() << " shutting down" << std::endl;
    this->endCapture();
}
