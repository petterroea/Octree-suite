#include <vector>

#include "../depthCamera/depthCamera.h"
#include "captureSettings.h"

class Capturer {
    std::vector<DepthCamera*> cameras;
    CaptureSettings settings;

public:
    Capturer(std::vector<DepthCamera*> cameras);

    void getFrame();
    void render(glm::mat4x4& view, glm::mat4x4& proj);
    void displayGui();
    void saveCalibration();
};
