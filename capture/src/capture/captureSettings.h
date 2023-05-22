#include <glm/mat4x4.hpp>

#include <vector>

#include <lineRenderer.h>

class CaptureSettings {
    glm::vec3 capturePosition;
    float captureScale = 1.0f;

    LineRendererShader lineRendererShader;
    LineRenderer lineRenderer;
public:
    CaptureSettings();

    void displayGui();
    void renderHelpLines(glm::mat4& view, glm::mat4& projection);

    glm::vec3 getCapturePosition() { return capturePosition; }
    float getCaptureScale() { return captureScale; }
};