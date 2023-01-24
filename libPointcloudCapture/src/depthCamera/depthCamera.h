#pragma once

#include <GL/glew.h>
#include <pthread.h>
#include <semaphore.h>
#include <string>

#include <librealsense2/rs.hpp>

#include <glm/mat4x4.hpp>

#include <cuda_runtime.h>

#include "types.h"
#include "../render/texturedPointcloudRenderer.h"

#include "gpuPointTransformer.h"

#include "../cameraCalibrator.h"

class DepthCamera {
private:
    // Thread functions
    pthread_t hThread;
    static void threadEntrypoint(DepthCamera* me);


    //Signalling
    sem_t frameRequestSemaphore;
    sem_t frameReceivedSemaphore;
protected:
    CameraCalibrator* cameraCalibrator;
    // Stores state relating to calibrating the current depth camera, and printing debug info from it
#ifndef HEADLESS_RELEASE
    TexturedPointcloudRenderer* renderer = nullptr;
#endif
    VideoMode videoMode;
    RenderMode renderMode;

    // Start the capture thread
    void startCaptureThread();
    // Thread functions
    void processingThread();
    virtual void processFrame() = 0;
    virtual void postCaptureCleanup() = 0;

    bool running = true;

    // Variables that aren't thread safe
    glm::mat4 calibratedTransform; // Model view matrix calibrated from OpenCV
    
    // Cuda stuff
#ifndef HEADLESS_RELEASE
    void setupGpuMemoryOpenGL(VideoMode mode);
#endif
    void setupGpuMemoryHeadless(VideoMode mode);

#ifndef HEADLESS_RELEASE
    void mapGlTextureToCudaArray(GLuint glTexture, cudaArray_t* cudaArray);
    void mapGlBufferToCuda(GLuint glTexture, void** devPtr);
#endif

    // Texture 
    cudaArray_t cuArrayTexRgba = nullptr;
    cudaSurfaceObject_t cuSurfaceObjectTexRgba = 0;
    cudaTextureObject_t cuTextureObjectRgba = 0;

    // Buffers
    void* devPtrPoints = nullptr;
    void* devPtrTexCoords = nullptr;

    // Container for handling double-buffered vector transformation on GPU
    GpuPointTransformer gpuTransformer;

    // Only for visualisation. Maybe we can ignore?
    //cudaArray_t cuArrayTexDepth = nullptr;
    //cudaSurfaceObject_t cuSurfaceObjectTexDepth = 0;
    int pointCount = 0;

public:
    DepthCamera(CameraCalibrator* cameraCalibrator, RenderMode renderMode, VideoMode videoMode);
    ~DepthCamera();

    //const rs2::vertex* getVertices() { return (const rs2::vertex*)this->lastVertices; }
    //const rs2::texture_coordinate* getTextureCoordinates() { return (const rs2::texture_coordinate*) this->lastTextureCoordinates; }

    glm::mat4& getCalibration() { return this->calibratedTransform; }
    void setCalibration(glm::mat4 newCalibration) { this->calibratedTransform = newCalibration; }

    // Start functions
    virtual void beginStreaming() = 0;
    virtual void beginRecording(const std::string filename) = 0;

    // Thread shutdown
    void endCaptureThread();
    void waitForThreadJoin();
    // Start frame fetching in another thread
    bool requestFrame();
    // Wait for frame fetch to finish
    void waitForNewFrame();

    int getPointCount() { return this->pointCount; };
    void capturePoints(glm::vec3** points, glm::vec3** colors, int* count, glm::mat4x4 captureTransform);

#ifndef HEADLESS_RELEASE
    TexturedPointcloudRenderer* getRenderer() { return this->renderer; }
#endif

    virtual void uploadGpuDataSync() = 0;
    virtual std::string getSerial() = 0;
    virtual std::string getKind() = 0;
};
