#include "../depthCamera.h"

class RealsenseDepthCamera : public DepthCamera{
    rs2::device device;
    rs2::config config;
    rs2::pointcloud pointcloud;

    rs2::vertex* lastVertices = nullptr;
    rs2::texture_coordinate* lastTextureCoordinates = nullptr;

    rs2::pipeline capturePipeline;

    rs2::frameset lastFrame;
    rs2::points lastPointcloud;

    //Thread functions
    virtual void processFrame();
    virtual void beginCapture();
    virtual void endCapture();

    // Need to convert image data to RGBA before uploading it to OpenGL
    unsigned char* textureConversionBuffer;

    void uploadToGpu();
public:
    RealsenseDepthCamera(RenderMode renderMode, rs2::device device, bool master);
    ~RealsenseDepthCamera();

    std::string getSerial();
    std::string getKind();
    void uploadGpuDataSync();
};