#include <vector>
#include <chrono>

#include "../depthCamera/depthCamera.h"
#include "captureSettings.h"
#include "asyncPointcloudWriter.h"

class Capturer {
    std::vector<DepthCamera*> cameras;
    CaptureSettings settings;
    AsyncPointcloudWriter writer;

    bool autoCalibrate = false;
    int autoCalibrateTreshold = 45;

    //Video-related stuff
    bool videoCapture = false;
    int framesCaptured = 0;
    std::chrono::time_point<std::chrono::system_clock> captureStart;

public:
    Capturer(std::vector<DepthCamera*> cameras);

    void getFrame();
    void render(glm::mat4x4& view, glm::mat4x4& proj);
    void displayGui();

    void saveCalibration();
    void loadCalibration();

    inline bool isCapturingVideo() { return this->videoCapture; };

    void capture();
};
