#include "octreeCapture.h"

#include "imgui.h"

#include <glm/gtc/matrix_transform.hpp>

OctreeCapture::OctreeCapture(): capturePosition(0.0f, 0.0f, 0.0f), lineRendererShader(), lineRenderer(&this->lineRendererShader) {
    // Axis lines
    this->lineRenderer.drawLine(
        glm::vec3(0.0f, 0.0f, 0.0f), 
        glm::vec3(1.0f, 0.0f, 0.0f), 
        glm::vec3(1.0f, 0.0f, 0.0f)
    );
    this->lineRenderer.drawLine(
        glm::vec3(0.0f, 0.0f, 0.0f), 
        glm::vec3(0.0f, 1.0f, 0.0f), 
        glm::vec3(0.0f, 1.0f, 0.0f)
    );
    this->lineRenderer.drawLine(
        glm::vec3(0.0f, 0.0f, 0.0f), 
        glm::vec3(0.0f, 0.0f, 1.0f), 
        glm::vec3(0.0f, 0.0f, 1.0f)
    );
    // Bottom box
    this->lineRenderer.drawLine(
        glm::vec3(-1.0f, 1.0f, -1.0f), 
        glm::vec3(-1.0f, 1.0f, 1.0f), 
        glm::vec3(1.0f, 1.0f, 1.0f)
    );
    this->lineRenderer.drawLine(
        glm::vec3(-1.0f, 1.0f, 1.0f), 
        glm::vec3(1.0f, 1.0f, 1.0f), 
        glm::vec3(1.0f, 1.0f, 1.0f)
    );
    this->lineRenderer.drawLine(
        glm::vec3(1.0f, 1.0f, 1.0f), 
        glm::vec3(1.0f, 1.0f, -1.0f), 
        glm::vec3(1.0f, 1.0f, 1.0f)
    );
    this->lineRenderer.drawLine(
        glm::vec3(1.0f, 1.0f, -1.0f), 
        glm::vec3(-1.0f, 1.0f, -1.0f), 
        glm::vec3(1.0f, 1.0f, 1.0f)
    );
    // Top box
    this->lineRenderer.drawLine(
        glm::vec3(-1.0f, -1.0f, -1.0f), 
        glm::vec3(-1.0f, -1.0f, 1.0f), 
        glm::vec3(1.0f, 1.0f, 1.0f)
    );
    this->lineRenderer.drawLine(
        glm::vec3(-1.0f, -1.0f, 1.0f), 
        glm::vec3(1.0f, -1.0f, 1.0f), 
        glm::vec3(1.0f, 1.0f, 1.0f)
    );
    this->lineRenderer.drawLine(
        glm::vec3(1.0f, -1.0f, 1.0f), 
        glm::vec3(1.0f, -1.0f, -1.0f), 
        glm::vec3(1.0f, 1.0f, 1.0f)
    );
    this->lineRenderer.drawLine(
        glm::vec3(1.0f, -1.0f, -1.0f), 
        glm::vec3(-1.0f, -1.0f, -1.0f), 
        glm::vec3(1.0f, 1.0f, 1.0f)
    );
    // Connecting lines
    this->lineRenderer.drawLine(
        glm::vec3(-1.0f, -1.0f, -1.0f), 
        glm::vec3(-1.0f, 1.0f, -1.0f), 
        glm::vec3(1.0f, 1.0f, 1.0f)
    );
    this->lineRenderer.drawLine(
        glm::vec3(-1.0f, -1.0f, 1.0f), 
        glm::vec3(-1.0f, 1.0f, 1.0f), 
        glm::vec3(1.0f, 1.0f, 1.0f)
    );
    this->lineRenderer.drawLine(
        glm::vec3(1.0f, -1.0f, -1.0f), 
        glm::vec3(1.0f, 1.0f, -1.0f), 
        glm::vec3(1.0f, 1.0f, 1.0f)
    );
    this->lineRenderer.drawLine(
        glm::vec3(1.0f, -1.0f, 1.0f), 
        glm::vec3(1.0f, 1.0f, 1.0f), 
        glm::vec3(1.0f, 1.0f, 1.0f)
    );
}

void OctreeCapture::displayGui() {
    ImGui::Begin("Capture settings");
    if(ImGui::CollapsingHeader("Bounds")) {
        ImGui::SliderFloat3("Offset", (float*)&this->capturePosition, -3.0f, 3.0f);
        ImGui::SliderFloat("Scale", &this->captureScale, 0.1f, 3.0f);
    }
    ImGui::Separator();
    if(ImGui::Button("Capture a frame")) {

    }

    ImGui::End();
}

void OctreeCapture::renderHelpLines(glm::mat4& view, glm::mat4& projection) {
    glm::mat4 model = glm::translate(glm::scale(glm::mat4x4(1.0f), glm::vec3(this->captureScale, this->captureScale, this->captureScale)), this->capturePosition);

    this->lineRenderer.setModelTransform(model);
    this->lineRenderer.render(view, projection);
}