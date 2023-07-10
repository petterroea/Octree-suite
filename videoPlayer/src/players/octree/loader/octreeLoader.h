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
    bool threadRunning = false;
    std::mutex ioMutex;

    void startLoadingThread();

    static void worker(OctreeLoader* me);

    unsigned char* readCompressedImage(int* bufferLength, std::ifstream& file);
    unsigned char* readCompressed(int* bufferLength, std::ifstream& file);
    void loadLayer(int layer, std::vector<int>& layerSizes, LayeredOctreeContainerStatic<octreeColorType>* container, std::ifstream& file);
    void calculateLeafFlags(int layer, int index, LayeredOctreeContainerStatic<octreeColorType>* container);
public:
    OctreeLoader(LayeredOctreeAllocator* allocator);
    ~OctreeLoader();

    loadedOctreeType* getLoadedOctree(OctreeFrameset** frameset);
    OctreeFrameset* peekLoadedOctreeFrameset();

    void requestFrameset(OctreeFrameset* frameset);

    OctreeFrameset* getCurrentlyLoadingFrameset();
    OctreeFrameset* getNextLoadingFrameset();

    // Expose some state i guess
    bool isLoadingThreadRunning();
};