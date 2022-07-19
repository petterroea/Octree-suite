#include "capturer.h"

#include <imgui.h>

Capturer::Capturer(std::vector<DepthCamera*> cameras) : cameras(cameras) {

}

void Capturer::getFrame() {
    for(auto device : this->cameras) {
        device->requestFrame();
    }
    for(auto device : this->cameras) {
        device->waitForNewFrame();
    }
    //Optional
    for(auto device : this->cameras) {
        device->uploadGpuDataSync();
    }
}

void Capturer::saveCalibration() {

}

void Capturer::render(glm::mat4x4& view, glm::mat4x4& proj) {
    for(auto device : this->cameras) {
        device->getRenderer()->render(device->getCalibration(), view, proj, device->getPointCount());
    }
    this->settings.renderHelpLines(view, proj);
}

void Capturer::displayGui() {
    ImGui::Begin("Capture");
    this->settings.displayGui();
    ImGui::End();
    ImGui::Begin("Devices");
    for(auto device : this->cameras) {
        ImGui::PushID(device->getSerial().c_str());

        char label[200];
        sprintf(label, "%s: %s", device->getKind().c_str(), device->getSerial().c_str());
        if(ImGui::CollapsingHeader(label)) {
            device->drawImmediateGui();
            ImGui::Separator();
        }
        ImGui::PopID();
    }
    ImGui::End();
}