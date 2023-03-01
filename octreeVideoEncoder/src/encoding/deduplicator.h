#pragma once

#include <thread>
#include <vector>
#include <mutex>

#include "octreeHashmap.h"
#include "../layeredOctree/layeredOctreeContainer.h"

struct DeDuplicationJob {
    int jobId;
    size_t count;
};

class DeDuplicator {
    int nThreads;
    std::vector<std::thread*> threadPool;

    std::mutex jobMutex;
    std::vector<DeDuplicationJob*> jobs;

    int layer;
    OctreeHashmap& hashmap;
    LayeredOctreeContainer<glm::vec3>& container;

    static void worker(DeDuplicator* me);
    std::vector<DeDuplicationJob*>::iterator currentJobIterator;
    DeDuplicationJob* getNextJob();

    // DeDuplication implementation
    void kMeans(int key, int k, int steps);
public:
    DeDuplicator(OctreeHashmap& hashmap, LayeredOctreeContainer<glm::vec3>& container, int layer, int nThreads);
    ~DeDuplicator();

    void run();
};