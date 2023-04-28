#pragma once

#include <thread>
#include <mutex>

#include <glm/vec3.hpp>

#include <layeredOctree/layeredOctreeContainerStatic.h>

#include "../octreeFrameset.h"
#include "../config.h"
#include "../layeredOctreeAllocator.h"

class OctreeLoader {
    OctreeFrameset* requestedFrameset = nullptr;
    OctreeFrameset* nextRequestedFrameset = nullptr;

    loadedOctreeType* loadedOctree = nullptr;
    OctreeFrameset* loadedFrameset = nullptr;

    LayeredOctreeAllocator* allocator;

    OctreeFrameset* lastLoadedFrameset = nullptr;
    LayeredOctreeContainerCuda<octreeColorType>* octreeContainer;

    std::thread* loadingThread = nullptr;
    std::mutex ioMutex;

    void startLoadingThread();
    bool isLoadingThreadRunning();

    static void worker(OctreeLoader* me);

public:
    OctreeLoader(LayeredOctreeAllocator* allocator);
    ~OctreeLoader();

    loadedOctreeType* getLoadedOctree(OctreeFrameset** frameset);

    void requestFrameset(OctreeFrameset* frameset);

    OctreeFrameset* getCurrentlyLoadingFrameset();
    OctreeFrameset* getNextLoadingFrameset();
};