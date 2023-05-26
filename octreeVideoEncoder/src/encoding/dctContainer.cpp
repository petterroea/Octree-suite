#include "dctContainer.h"

#include <fstream>

#include <dct/rle.h>

#include "dctEncoder.h"


DctContainer::~DctContainer() {
    for(auto layer : this->yLayers) {
        delete[] layer;
    }
    for(auto layer : this->uLayers) {
        delete[] layer;
    }
    for(auto layer : this->vLayers) {
        delete[] layer;
    }
}

void DctContainer::addYLayer(int idx, int size) {
    this->yLayerPositions.push_back(idx);
    this->yLayerSizes.push_back(size);
}

void DctContainer::addUVLayer(int idx, int size) {
    this->uvLayerPositions.push_back(idx);
    this->uvLayerSizes.push_back(size);
}

void DctContainer::build(LayeredOctreeProcessingContainer<octreeColorType>* container, int concurrency) {
    for(auto layer : this->yLayerPositions) {
        this->yLayers.push_back(this->buildLayer(layer, 0, container, concurrency));
    }
    for(auto layer : this->uvLayerPositions) {
        this->uLayers.push_back(this->buildLayer(layer, 1, container, concurrency));
        this->vLayers.push_back(this->buildLayer(layer, 2, container, concurrency));
    }
}

int* DctContainer::buildLayer(int layer, int colorIdx, LayeredOctreeProcessingContainer<octreeColorType>* container, int concurrency) {
    int layerSize = container->getLayerSize(layer);

    std::cout << "Creating quantized DCT table: from layer " << layer << " - " << layerSize << ", nodes, " << (DCT_TABLE_SIZE * layerSize) << " bytes." << std::endl;

    int* out = new int[DCT_TABLE_SIZE * layerSize];

    DctEncoder encoder(out, layer, layerSize, 5000, container, colorIdx);
    encoder.run(concurrency);

    return out;
}

void DctContainer::serialize(std::ofstream* out) {
    // Write header
    short layerCount = this->yLayerSizes.size();
    out->write((char*)&layerCount, sizeof(layerCount));
    layerCount = this->uvLayerSizes.size();
    out->write((char*)&layerCount, sizeof(layerCount));
    
    // Write Y layers
    for(int i = 0; i < this->yLayerSizes.size(); i++) {
        auto layer = this->yLayers[i];
        auto layerSize = this->yLayerSizes[i];
        write_run_time_length(layer, DCT_TABLE_SIZE * layerSize, out);
    }

    // Write U layer
    for(int i = 0; i < this->uvLayerSizes.size(); i++) {
        auto uLayer = this->uLayers[i];
        auto vLayer = this->vLayers[i];

        auto layerSize = this->uvLayerSizes[i];
        write_run_time_length(uLayer, DCT_TABLE_SIZE * layerSize, out);
        write_run_time_length(vLayer, DCT_TABLE_SIZE * layerSize, out);
    }
}