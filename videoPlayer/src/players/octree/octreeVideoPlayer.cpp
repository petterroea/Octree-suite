#include "octreeVideoPlayer.h"

OctreeVideoPlayer::OctreeVideoPlayer(TimeProvider* timeProvider, std::filesystem::path videoPath) : 
    timeProvider(timeProvider),
    videoPath(videoPath), 
    metadata(videoPath),
    loader(&this->allocator) {
    OctreeFrameset* firstFrame = this->metadata.getFramesetByFrame(0);
    this->loader.requestFrameset(firstFrame);
    std::cout << "LOADER: Requested the first frame" << std::endl;

    this->renderer = new OctreeRenderer();
}

OctreeVideoPlayer::~OctreeVideoPlayer() {
    delete this->renderer;
}

void OctreeVideoPlayer::render(int width, int height, glm::mat4 view, glm::mat4 projection) {
    this->renderer->updateTexture(width, height);
    // Figure out what frameste to render
    if(this->currentFrame == nullptr) {
        // Check if a frame is loaded
        auto loadedFrame = this->loader.getLoadedOctree(&this->currentFrameset);
        if(loadedFrame != nullptr) {
            this->currentFrame = loadedFrame;
        } else {
            // Nothing to render, return
            std::cout << "No frame loaded" << std::endl;
            return;
        }
    }
    
    // Figure out if we need to order a new frameset from the octreeloader.
    // Get currently loading
    // Get next in line
    // If not? Request next frameset

    // Figure out what frame to render
    int currentFrame = this->getCurrentFrame();
    int renderFrame = 0;
    if(currentFrame > this->currentFrameset->getEndIndex()) {
        // If we are past the last frame, render the last frame
        renderFrame = this->currentFrameset->getEndIndex() - this->currentFrameset->getStartIndex();
    } else if(currentFrame < this->currentFrameset->getStartIndex()) {
        renderFrame = 0;
    } else {
        renderFrame = currentFrame - this->currentFrameset->getStartIndex();
    }

    // We know what frame to render, now render it
    this->renderer->render(this->currentFrame, renderFrame, view, projection);
}

float OctreeVideoPlayer::getVideoLength() {
    return 10.0f;
}

float OctreeVideoPlayer::getFps() {
    return 30.0f;
}

int OctreeVideoPlayer::getCurrentFrame() {
    float time = this->timeProvider->getTime();
    float fps = this->getFps();
    return static_cast<int>(time * fps);
}

bool OctreeVideoPlayer::isBuffering() {
    bool buffering = this->currentFrame == nullptr;
    std::cout << "Buffering? " << (buffering ? "yes" : "no") << std::endl;
    return buffering;
}