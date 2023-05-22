#include "plyVideoPlayer.h"

#include <glm/mat4x4.hpp>

#include <iostream>

PlyVideoPlayer::PlyVideoPlayer(TimeProvider* timeProvider, const std::filesystem::path folder) : 
    VideoPlayer(timeProvider),
    metadata(folder) {
    this->loader = new PlyLoader(&this->metadata);
    this->loader->startLoading(0);
    this->renderer = new PlyRenderer();
}

PlyVideoPlayer::~PlyVideoPlayer() {
    delete this->loader;
}

void PlyVideoPlayer::render(int width, int height, glm::mat4 view, glm::mat4 projection) {
    // Figure out what frame we are on
    int animationFrame = this->getCurrentFrame();
    // Beginning of render case
    if(!currentRenderingPointcloud) {
        int loadedFrame = -1;
        auto loadedPointcloud = this->loader->getNextFrame(&loadedFrame);
        if(!loadedPointcloud) {
            std::cout << "No ply to render yet" << std::endl;
            return;
        }
        this->setNewFrame(loadedPointcloud, loadedFrame);
    }
    else if(animationFrame > this->currentRenderingFrame) {
        // If we have progressed in time, try to find the next frame in the queue
        this->seekPlyFrame();
    } else if(animationFrame < this->currentRenderingFrame) {
        // The user has seeked backwards in time, request a full re-load
        std::cout << "Backwards seek detected, (" << animationFrame << " < " << this->currentRenderingFrame << ")requesting new re-load" << std::endl;
        if(lastRequestedSeek > animationFrame) {
            this->loader->startLoading(animationFrame);
            this->lastRequestedSeek = animationFrame;
        } else {
            std::cout << "We have already requested a prior seek, being patient..." << std::endl;
            this->seekPlyFrame();
        }
    }
    // Render whatever we have
    glm::mat4 model(1.0f);
    this->renderer->render(model, view, projection);
}

void PlyVideoPlayer::seekPlyFrame() {
    int animationFrame = this->getCurrentFrame();
    std::cout << "START SEEK, current frame: " << animationFrame << std::endl;

    int loadedFrame = -1;
    auto loadedPointcloud = this->loader->getNextFrame(&loadedFrame);
    std::cout << "SEEK: " << loadedFrame << std::endl;;
    while(loadedFrame != animationFrame && loadedPointcloud != nullptr) {
        delete loadedPointcloud;
        auto loadedPointcloud = this->loader->getNextFrame(&loadedFrame);
        std::cout << "SEEK: " << loadedFrame << std::endl;;
    }
    if(loadedFrame != animationFrame) {
        std::cout << "WARNING: Tried to seek to current frame but failed" << std::endl;
    } else {
        std::cout << "We good" << std::endl;
        this->setNewFrame(loadedPointcloud, loadedFrame);
    }
}

void PlyVideoPlayer::setNewFrame(Pointcloud* pointcloud, int frame) {
    if(this->currentRenderingPointcloud) {
        delete this->currentRenderingPointcloud;
    }
    this->currentRenderingFrame = frame;
    this->currentRenderingPointcloud = pointcloud;
    this->renderer->uploadFrame(
        pointcloud->getVertices(), 
        pointcloud->getColors(), 
        pointcloud->getPointCount()
    );
}


float PlyVideoPlayer::getVideoLength() {
    return static_cast<float>(this->metadata.getFrameCount()) / this->metadata.getFps();
}

bool PlyVideoPlayer::isBuffering() {
    return false; //ToDO
}

void PlyVideoPlayer::getVideoMetadata(VideoMetadata* metadata) const {
    metadata->fps = this->metadata.getFps();
    metadata->frameCount = this->metadata.getFrameCount();
}