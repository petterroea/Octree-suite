#include <GL/glew.h>
#include <pthread.h>

#include <librealsense2/rs.hpp>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/aruco.hpp>

#include <glm/mat4x4.hpp>

class DepthCamera {
    GLuint colorTexture;
    GLuint depthTexture;

    rs2::device device;
    rs2::config config;
    rs2::pointcloud pointcloud;

    rs2::vertex* lastVertices = nullptr;
    rs2::texture_coordinate* lastTextureCoordinates = nullptr;
    int lastVertexCount = 0;

    rs2::pipeline capturePipeline;

    // The abillity to enable and disable OpenCV activities
    bool openCvEnabled = false;
    glm::mat4 calibratedTransform;
    float cameraOffset = 0.0f;

    //static void depthCameraThread(DepthCamera* camera);
    static GLuint buildTexture();
public:
    DepthCamera(rs2::device device);
    ~DepthCamera();

    GLuint getColorTextureHandle() { return this->colorTexture; }
    GLuint getDepthTextureHandle() { return this->depthTexture; }

    const rs2::vertex* getVertices() { return (const rs2::vertex*)this->lastVertices; }
    const rs2::texture_coordinate* getTextureCoordinates() { return (const rs2::texture_coordinate*) this->lastTextureCoordinates; }

    rs2::frameset processFrame();
    void uploadTextures(rs2::frameset& frame);
    rs2::points processPointcloud(rs2::frameset& frame);
    glm::mat4& getCalibration() { return this->calibratedTransform; }

    void begin();
    void end();
    inline std::string getSerial() { return this->device.get_info(rs2_camera_info::RS2_CAMERA_INFO_SERIAL_NUMBER); }
};