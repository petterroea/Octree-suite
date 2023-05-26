#pragma once

#include <vector>

#include "../structures/layeredOctreeProcessingContainer.h"

class DctContainer {   
    std::vector<int> yLayerPositions;
    std::vector<int> uvLayerPositions;

    std::vector<int> yLayerSizes;
    std::vector<int> uvLayerSizes;

    std::vector<int*> yLayers;
    std::vector<int*> uLayers;
    std::vector<int*> vLayers;

    int* buildLayer(int layer, int colorIdx, LayeredOctreeProcessingContainer<octreeColorType>* container, int concurrency);

public:
    ~DctContainer();

    void addYLayer(int idx, int size);
    void addUVLayer(int idx, int size);
    void build(LayeredOctreeProcessingContainer<octreeColorType>* container, int concurrency);

    void serialize(std::ofstream* out);
    void deserialize(std::ifstream* in);
};