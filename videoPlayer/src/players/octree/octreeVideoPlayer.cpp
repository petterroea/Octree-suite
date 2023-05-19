#include "octreeVideoPlayer.h"

#include "imgui.h"

OctreeVideoPlayer::OctreeVideoPlayer(TimeProvider* timeProvider, std::filesystem::path videoPath) : 
    VideoPlayer(timeProvider),
    videoPath(videoPath), 
    metadata(videoPath),
    loader(&this->allocator) {
    OctreeFrameset* firstFrame = this->metadata.getFramesetByFrame(0);
    // Add some frames to the queue
    this->loader.requestFrameset(firstFrame);
    auto nextFrame = this->metadata.getFramesetByFrame(firstFrame->getEndIndex() + 1);
    this->loader.requestFrameset(nextFrame);
    std::cout << "LOADER: Requested the first frame" << std::endl;

    this->renderer = new OctreeRenderer();
}

OctreeVideoPlayer::~OctreeVideoPlayer() {
    delete this->renderer;
}

void OctreeVideoPlayer::getVideoMetadata(VideoMetadata* metadata) const {
    metadata->fps = this->metadata.getFps();
    metadata->frameCount = this->metadata.getFrameCount();
}

void OctreeVideoPlayer::drawDebugInfo() {
    ImGui::Begin("OctreeVideoPlayer state");
    ImGui::Text("Frame: %d", this->getCurrentFrame());
    if(this->currentFrameset) {
        ImGui::Text("Current frameset: %d -> %d", this->currentFrameset->getStartIndex(), this->currentFrameset->getEndIndex());
    } else {
        ImGui::Text("Current frameset: None");
    }
    auto nextLoaded = this->loader.peekLoadedOctreeFrameset();
    if(nextLoaded) {
        ImGui::Text("Loaded and ready: %d -> %d", nextLoaded->getStartIndex(), nextLoaded->getEndIndex());
    } else {
        ImGui::Text("Loaded and ready: None");
    }

    ImGui::Separator();

    auto currentlyLoading = this->loader.getCurrentlyLoadingFrameset();
    if(currentlyLoading) {
        ImGui::Text("Currently loading: %d -> %d", currentlyLoading->getStartIndex(), currentlyLoading->getEndIndex());
    } else {
        ImGui::Text("Currently loading: None");
    }
    auto nextInLine = this->loader.getNextLoadingFrameset();
    if(nextInLine) {
        ImGui::Text("Next in line: %d -> %d", nextInLine->getStartIndex(), nextInLine->getEndIndex());
    } else {
        ImGui::Text("Next in line: None");
    }
    ImGui::End();
}

void OctreeVideoPlayer::render(int width, int height, glm::mat4 view, glm::mat4 projection) {
    this->renderer->updateTexture(width, height);
    // Show debug info
    this->drawDebugInfo();

    // Figure out what frame to render
    if(this->currentFrame == nullptr) {
        // Check if a frame is loaded
        auto loadedFrame = this->loader.getLoadedOctree(&this->currentFrameset);
        if(loadedFrame != nullptr) {
            this->currentFrame = loadedFrame;
            // Ask for the next frame to be loaded immediately
            OctreeFrameset* nextFrameset = this->metadata.getFramesetByFrame(this->currentFrameset->getEndIndex()+1);
            this->loader.requestFrameset(nextFrameset);
        } else {
            // Nothing to render, return
            std::cout << "No frame loaded" << std::endl;
            return;
        }
    } else {
        std::cout << "Current frameset: " << this->currentFrameset->getStartIndex() << " -> " << this->currentFrameset->getEndIndex() << std::endl;
    }
    
    // Figure out if we need to order a new frameset from the octreeloader.
    // Get currently loading
    // Get next in line
    // If not? Request next frameset

    // Figure out what frame to render
    int currentFrameIndex = this->getCurrentFrame();
    int renderFrame = 0;
    if(currentFrameIndex > this->currentFrameset->getEndIndex()) {
        std::cout << "Past the current frameset" << std::endl;
        // We are past the last frame of the current frameset, see if we are within the next frameset
        OctreeFrameset* nextFrameset = nullptr;
        auto nextFrame = this->loader.getLoadedOctree(&nextFrameset);
        // There may be unwanted frames in the encoding queue, 
        // so iterate until we run out of loaded frames or find the correct one
        while(nextFrame != nullptr) { 
            if(currentFrameIndex >= nextFrameset->getStartIndex() && currentFrameIndex <= nextFrameset->getEndIndex()) {
                // We found the frame we wanted 
                std::cout << "Next frame was in the queue" << std::endl;
                break;
            } else {
                std::cout << "Deleting frame in queue that wasn't wanted" << std::endl;
                // The user seeked, so the current frame is not the one we want to show at the moment.
                delete nextFrame;
            }
            nextFrame = this->loader.getLoadedOctree(&nextFrameset);
        }
        // If the next frameset is loaded, move to it
        if(nextFrame) {
            std::cout << "Moving to current frame" << std::endl;
            delete this->currentFrame;
            this->currentFrame = nextFrame;
            this->currentFrameset = nextFrameset;

            renderFrame = currentFrameIndex - this->currentFrameset->getStartIndex();

            // Request the next frame
            // Determine the next frameset we want
            OctreeFrameset* nextFramesetInLine = this->metadata.getFramesetByFrame(this->currentFrameset->getEndIndex()+1);
            if(nextFramesetInLine == nullptr) {
                std::cout << "Reached end of video" << std::endl;
            } else {
                // Request the next frame in line if it isn't already requested
                if(
                    this->loader.getCurrentlyLoadingFrameset() != nextFramesetInLine &&
                    this->loader.getNextLoadingFrameset() != nextFramesetInLine
                ) {
                    this->loader.requestFrameset(nextFramesetInLine);
                }
            }
        } else {
            std::cout << "Next frame is not ready yet" << std::endl;
            // If we are past the last frame, render the last frame
            renderFrame = this->currentFrameset->getEndIndex() - this->currentFrameset->getStartIndex();
        }

    } else if(currentFrameIndex < this->currentFrameset->getStartIndex()) {
        std::cout << "Before the current frameset" << std::endl;
        renderFrame = 0;
    } else {
        std::cout << "In correct frameset" << std::endl;
        renderFrame = currentFrameIndex - this->currentFrameset->getStartIndex();
    }

    // We know what frame to render, now render it
    this->renderer->render(this->currentFrame, renderFrame, view, projection);
}

float OctreeVideoPlayer::getVideoLength() {
    return 10.0f;
}

bool OctreeVideoPlayer::isBuffering() {
    bool buffering = this->currentFrame == nullptr;
    std::cout << "Buffering? " << (buffering ? "yes" : "no") << std::endl;
    return buffering;
}