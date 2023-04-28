#pragma once

#include <thread>
#include <vector>
#include <mutex>

#include "octreeHashmap.h"
#include "../config.h"
#include "../videoEncoderRunArgs.h"
#include "../structures/octreeProcessingPayload.h"
#include "../structures/layeredOctreeProcessingContainer.h"
#include <layeredOctree/layeredOctreeContainerCuda.h>

struct DeDuplicationJob {
    int jobId;
    size_t count;
};

class DeDuplicator {
    VideoEncoderRunArgs* args;
    std::vector<std::thread*> threadPool;

    std::mutex jobMutex;
    std::vector<DeDuplicationJob*> jobs;

    int layer;
    OctreeHashmap& hashmap;
    LayeredOctreeProcessingContainer<octreeColorType>& container;
    //LayeredOctreeContainerCuda<OctreeProcessingPayload<octreeColorType>>* cudaContainer;

    static void worker(DeDuplicator* me);
    std::vector<DeDuplicationJob*>::iterator currentJobIterator;
    DeDuplicationJob* getNextJob();

    // DeDuplication implementation
    void kMeans(int key, int k, int steps);
    void markTreeAsTrimmed(int layer, int index);
public:
    DeDuplicator(OctreeHashmap& hashmap, LayeredOctreeProcessingContainer<octreeColorType>& container, int layer, VideoEncoderRunArgs* args);
    ~DeDuplicator();

    void run();
};