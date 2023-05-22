#include "captureSettings.h"

#include <glm/gtc/matrix_transform.hpp>

#include <imgui.h>

CaptureSettings::CaptureSettings(): capturePosition(0.0f, 0.0f, 0.0f), lineRendererShader(), lineRenderer(&this->lineRendererShader) {
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

void CaptureSettings::displayGui() {
    if(ImGui::CollapsingHeader("Bounds")) {
        ImGui::SliderFloat3("Offset", (float*)&this->capturePosition, -1.5f, 1.5f);
        ImGui::SliderFloat("Scale", &this->captureScale, 0.1f, 2.5f);
    }
}


void CaptureSettings::renderHelpLines(glm::mat4& view, glm::mat4& projection) {
    glm::mat4 model = glm::scale(glm::translate(glm::mat4x4(1.0f), this->capturePosition), glm::vec3(this->captureScale, this->captureScale, this->captureScale)) ;

    this->lineRenderer.setModelTransform(model);
    this->lineRenderer.render(view, projection);
}