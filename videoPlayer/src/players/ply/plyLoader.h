#pragma once

#include <queue>
#include <thread>
#include <mutex>

#include <pointcloud/pointcloud.h>

#include "plyMetadata.h"

class PlyLoader {
    int loadQueueSize = 30;

    PlyMetadata* metadata;
    int loadStart = 0;
    int loadCount = 0;
    std::queue<Pointcloud*> loadedPointclouds;
    std::queue<int> loadedFrames;

    std::thread* loadingThread = nullptr;
    bool threadRunning = false;
    bool shouldThreadRun = false;
    std::mutex mutex;

    void startLoadingThread();
    static void worker(PlyLoader* me);
    void workerInner();
    void stopLoadingThread();

    void wipeLoadedQueue();

public:
    PlyLoader(PlyMetadata* metadata);
    ~PlyLoader();

    void startLoading(int frame);

    Pointcloud* getNextFrame(int* loadedFrameNo);

};