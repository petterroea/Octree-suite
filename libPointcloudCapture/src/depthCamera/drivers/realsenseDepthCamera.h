#include "../depthCamera.h"

class RealsenseDepthCamera : public DepthCamera{
    rs2::device device;
    rs2::pointcloud pointcloud;

    rs2::vertex* lastVertices = nullptr;
    rs2::texture_coordinate* lastTextureCoordinates = nullptr;

    rs2::pipeline capturePipeline;

    rs2::frameset lastFrame;
    rs2::points lastPointcloud;

    //Thread functions
    virtual void processFrame();
    virtual void postCaptureCleanup();

    // Upload the RGB texture to this first, then
    // use a cuda kernel to add the alpha channel and
    // move the data to the RGBA texture defined by
    // our parent object, DepthCamera
    void* cuTexRgb;

    void uploadToGpu();
public:
    RealsenseDepthCamera(CameraCalibrator* cameraCalibrator, RenderMode renderMode, rs2::device device, bool master);
    ~RealsenseDepthCamera();

    //Startup functions
    virtual void beginStreaming();
    virtual void beginRecording(const std::string filename);

    std::string getSerial();
    std::string getKind();
    void uploadGpuDataSync();
};