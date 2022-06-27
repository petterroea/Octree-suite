#include <GL/glew.h>
#include <pthread.h>
#include <semaphore.h>

#include <librealsense2/rs.hpp>

#include <glm/mat4x4.hpp>

#include "openCVCalibrator.h"

class DepthCamera {
    GLuint colorTexture;
    GLuint depthTexture;

    rs2::device device;
    rs2::config config;
    rs2::pointcloud pointcloud;

    rs2::vertex* lastVertices = nullptr;
    rs2::texture_coordinate* lastTextureCoordinates = nullptr;
    int lastVertexCount = 0;

    // Stores state relating to calibrating the current depth camera, and printing debug info from it
    OpenCVCalibrator calibrator;
    bool calibrationEnabled = true;

    rs2::pipeline capturePipeline;


    //static void depthCameraThread(DepthCamera* camera);
    static GLuint buildTexture();

    // Thread functions
    pthread_t hThread;
    static void threadEntrypoint(DepthCamera* me);
    void processingThread();
    void processFrame();
    rs2::points processPointcloud(rs2::frameset& frame);

    //Signalling
    sem_t frameRequestSemaphore;
    sem_t frameReceivedSemaphore;
    bool running = true;

    // Variables that aren't thread safe
    glm::mat4 calibratedTransform; // Model view matrix calibrated from OpenCV
    rs2::frameset lastFrame;
    rs2::points lastPointcloud;
    
public:
    DepthCamera(rs2::device device, bool master);
    ~DepthCamera();

    GLuint getColorTextureHandle() { return this->colorTexture; }
    GLuint getDepthTextureHandle() { return this->depthTexture; }

    const rs2::vertex* getVertices() { return (const rs2::vertex*)this->lastVertices; }
    const rs2::texture_coordinate* getTextureCoordinates() { return (const rs2::texture_coordinate*) this->lastTextureCoordinates; }

    void uploadTextures();
    rs2::points getLastPointcloud();
    glm::mat4& getCalibration() { return this->calibratedTransform; }

    void drawImmediateGui();

    void begin();
    void end();
    void waitForThreadJoin();
    void requestFrame();
    void waitForNewFrame();

    inline std::string getSerial() { return this->device.get_info(rs2_camera_info::RS2_CAMERA_INFO_SERIAL_NUMBER); }
};