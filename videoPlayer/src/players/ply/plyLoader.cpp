#include "plyLoader.h"

#include <exception>
#include <iostream>

void PlyLoader::startLoadingThread() {
    std::cout << "Starting loading thread..." << std::endl;
    std::cout << "Mutex lock" << std::endl;
    if(this->threadRunning) {
        throw std::runtime_error("Tried to start thread, but thread is already running");
    }
    if(this->loadingThread) {
        std::cout << "Loading thread is already running, quitting it..." << std::endl;
        this->shouldThreadRun = false;
        this->loadingThread->join();
        delete this->loadingThread;
    }
    this->threadRunning = true;
    this->shouldThreadRun = true;
    this->loadingThread = new std::thread(PlyLoader::worker, this);
    std::cout << "Loading thread started..." << std::endl;
}

void PlyLoader::worker(PlyLoader* me) {
    me->workerInner();
}

void PlyLoader::workerInner() {
    std::cout << "Worker starting" << std::endl;
    this->mutex.lock();
    while(this->shouldThreadRun) {
        // See if there is more to load
        if(this->loadedPointclouds.size() > this->loadQueueSize) {
            this->shouldThreadRun = false;
            this->mutex.unlock();
            break;
        }
        int lastLoadStart = this->loadStart;
        int lastLoadCount = this->loadCount;
        int currentFrame = this->loadStart + this->loadCount;
        if(currentFrame >= this->metadata->getFrameCount()) {
            std::cout << "Reached the end of the sequence, stopping loading" << std::endl;
            this->shouldThreadRun = false;
            this->mutex.unlock();
            break;
        }
        std::cout << "Loading " << currentFrame << " from " << this->metadata->getPath() << std::endl;
        this->mutex.unlock();
        // Determine the filename
        char filename[256];
        sprintf(filename, "capture_%06d.ply", currentFrame);
        std::filesystem::path filePath = this->metadata->getPath() / std::filesystem::path(filename);
        if (!std::filesystem::exists(filePath)) {
            throw std::runtime_error("Ply file does not exist: " + filePath.string());
        }
        // Load the pointcloud
        Pointcloud* cloud = parsePlyFile(filePath.string());
        if(!cloud) {
            throw std::runtime_error("Unable to load, giving up");
        }
        this->mutex.lock();
        this->loadedPointclouds.push(cloud);
        this->loadedFrames.push(currentFrame);
        std::cout << "Successfully loaded frame " << std::to_string(currentFrame) << std::endl;
        this->loadCount++;
    }
    this->mutex.lock();
    this->threadRunning = false;
    this->mutex.unlock();
}

PlyLoader::PlyLoader(PlyMetadata* metadata) : metadata(metadata){

}

PlyLoader::~PlyLoader() {
    if(this->loadingThread) {
        this->shouldThreadRun = false;
        this->loadingThread->join();
        std::cout << "Shut down loading thread" << std::endl;
    }
    this->wipeLoadedQueue();
}

// Forces the list of loaded frames to empty, then starts loading again
void PlyLoader::startLoading(int frame) {
    std::cout << "Starting load" << std::endl;
    std::lock_guard<std::mutex> lock(this->mutex);
    this->loadStart = frame;
    this->loadCount = 0;
    this->wipeLoadedQueue();
    if(!this->threadRunning) {
        this->startLoadingThread();
    }
}

void PlyLoader::wipeLoadedQueue() {
    std::cout << "Wiping loaded frames" << std::endl;
    while(!this->loadedPointclouds.empty()) {
        delete this->loadedPointclouds.front();
        this->loadedPointclouds.pop();
        this->loadedFrames.pop();
    }
}

Pointcloud* PlyLoader::getNextFrame(int* loadedFrameNo) {
    std::cout << "Getting next frame" << std::endl;
    std::lock_guard<std::mutex> lock(this->mutex);
    // If we are able to load more frames and we aren't currently, star the loading thread
    if(this->loadedPointclouds.size() < this->loadQueueSize-1) {
        if(!this->threadRunning) {
            std::cout << "Re-starting loading thread" << std::endl;
            this->startLoadingThread();
        }
    }
    if(this->loadedPointclouds.empty()) {
        std::cout << "No loaded data available" << std::endl;
        return nullptr;
    }
    auto value = this->loadedPointclouds.front();
    this->loadedPointclouds.pop();
    *loadedFrameNo = this->loadedFrames.front();
    this->loadedFrames.pop();
    return value;
}
