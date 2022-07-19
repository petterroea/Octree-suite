#pragma once

#include <GL/glew.h>
#include <pthread.h>
#include <semaphore.h>

#include <librealsense2/rs.hpp>

#include <glm/mat4x4.hpp>

#include "openCVCalibrator.h"
#include <cuda_runtime.h>

#include "types.h"
#include "../render/texturedPointcloudRenderer.h"


class DepthCamera {
private:
    // Thread functions
    pthread_t hThread;
    static void threadEntrypoint(DepthCamera* me);

    //Signalling
    sem_t frameRequestSemaphore;
    sem_t frameReceivedSemaphore;
protected:
    // Stores state relating to calibrating the current depth camera, and printing debug info from it
    OpenCVCalibrator calibrator;
    bool calibrationEnabled = true;

    TexturedPointcloudRenderer* renderer = nullptr;
    VideoMode videoMode;
    RenderMode renderMode;

    void processingThread();
    virtual void processFrame() = 0;
    virtual void beginCapture() = 0;
    virtual void endCapture() = 0;

    bool running = true;

    // Variables that aren't thread safe
    glm::mat4 calibratedTransform; // Model view matrix calibrated from OpenCV
    
    // Cuda stuff
    void setupGpuMemoryOpenGL(VideoMode mode);
    void setupGpuMemoryHeadless(VideoMode mode);

    void mapGlTextureToCuda(GLuint glTexture, cudaArray_t* cudaArray, cudaSurfaceObject_t* surfaceObject);
    void mapGlBufferToCuda(GLuint glTexture, void** devPtr);
    
    cudaArray_t cuArrayTexRgb = nullptr;
    cudaSurfaceObject_t cuSurfaceObjectTexRgb = 0;

    void* devPtrPoints = nullptr;

    void* devPtrTexCoords = nullptr;

    // Only for visualisation. Maybe we can ignore?
    cudaArray_t cuArrayTexDepth = nullptr;
    cudaSurfaceObject_t cuSurfaceObjectTexDepth = 0;

    int pointCount = 0;

public:
    DepthCamera(RenderMode renderMode, VideoMode videoMode);
    ~DepthCamera();

    //const rs2::vertex* getVertices() { return (const rs2::vertex*)this->lastVertices; }
    //const rs2::texture_coordinate* getTextureCoordinates() { return (const rs2::texture_coordinate*) this->lastTextureCoordinates; }

    glm::mat4& getCalibration() { return this->calibratedTransform; }

    void drawImmediateGui();

    // Start the capture thread
    void startCaptureThread();
    // Thread shutdown
    void endCaptureThread();
    void waitForThreadJoin();
    // Start frame fetching in another thread
    void requestFrame();
    // Wait for frame fetch to finish
    void waitForNewFrame();

    int getPointCount() { return this->pointCount; };
    TexturedPointcloudRenderer* getRenderer() { return this->renderer; }

    virtual void uploadGpuDataSync() = 0;
    virtual std::string getSerial() = 0;
    virtual std::string getKind() = 0;
};