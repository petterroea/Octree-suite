#include <vector>

#include "../depthCamera/depthCamera.h"
#include "captureSettings.h"
#include "asyncPointcloudWriter.h"

class Capturer {
    std::vector<DepthCamera*> cameras;
    CaptureSettings settings;
    AsyncPointcloudWriter writer;

    bool autoCalibrate = false;
    int autoCalibrateTreshold = 45;

public:
    Capturer(std::vector<DepthCamera*> cameras);

    void getFrame();
    void render(glm::mat4x4& view, glm::mat4x4& proj);
    void displayGui();

    void saveCalibration();
    void loadCalibration();

    void capture();
};
