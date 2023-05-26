#pragma once
#include <thread>
#include <mutex>
#include <queue>

#include <dct/dct.h>

#include "../structures/layeredOctreeProcessingContainer.h"

#define DCT_TABLE_SIZE  (DCT_SIZE * DCT_SIZE * DCT_SIZE)

struct DctChildEntry {
    int index;
    int x;
    int y;
    int z;
};


class DctEncoder {
    int* layerPtr;
    int layerSize;
    int jobSize;
    int layer;

    int colorIdx;

    LayeredOctreeProcessingContainer<octreeColorType>* container;

    static void worker(DctEncoder* me);
    void workerInner();

    std::queue<int> jobs;
    int getJob();

    std::mutex jobMutex;

    void buildChildList(int layersToGo, DctChildEntry* childList, int* childCount, int layer, int index, int x, int y, int z);
public:
    DctEncoder(int* layerPtr, int layer, int layerSize, int jobSize, LayeredOctreeProcessingContainer<octreeColorType>* container, int colorIdx);
    void run(int paralellism);
};