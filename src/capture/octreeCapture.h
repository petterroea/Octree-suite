#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>

#include "../lib/lineRenderer.h"
// Responsible for keeping capture configuration as well as performing the actual capture
class OctreeCapture {
    glm::vec3 capturePosition;
    float captureScale = 1.0f;

    LineRendererShader lineRendererShader;
    LineRenderer lineRenderer;
public:
    OctreeCapture();
    void displayGui();
    void renderHelpLines(glm::mat4& view, glm::mat4& projection);
};