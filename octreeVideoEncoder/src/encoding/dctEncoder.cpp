#include "dctEncoder.h"

#include <thread>

#include <dct/quantization.h>

DctEncoder::DctEncoder(int* layerPtr, int layer, int layerSize, int jobSize, LayeredOctreeProcessingContainer<octreeColorType>* container, int colorIdx):
    layerPtr(layerPtr), 
    layer(layer),
    layerSize(layerSize), 
    jobSize(jobSize),
    container(container),
    colorIdx(colorIdx)
    {
    for(int i = 0; i < layerSize; i += jobSize) {
        jobs.push(i);
    }
}


void DctEncoder::worker(DctEncoder* me) {
    me->workerInner();
}

void DctEncoder::buildChildList(int layersToGo, DctChildEntry* childList, int* childCount, int layer, int index, int x, int y, int z) {
    //std::cout << "Building child list: " << layer << std::endl;
    if(layersToGo == 0) {
        //std::cout << "Hit bottom" << std::endl;
        auto node = this->container->getNode(layer, index);
        for(int i = 0; i < OCTREE_SIZE; i++) {
            int child = node->getChildByIdx(i);
            if(child == NO_NODE) {

            } else if(child < 0) {
                throw std::runtime_error("fuck: " + std::to_string(i) + ", " + std::to_string(index));

            } else if(child != NO_NODE) {
                int childCountIndex = (*childCount)++;

                if(childCountIndex > 8*8*8) {
                    std::cout << "Too many child counts: " << childCountIndex << std::endl;
                    throw std::runtime_error("Too many children");
                }

                childList[childCountIndex].index = child;
                childList[childCountIndex].x = (x << 1) | OCTREE_EXTRACT_X(i);
                childList[childCountIndex].y = (y << 1) | OCTREE_EXTRACT_Y(i);
                childList[childCountIndex].z = (z << 1) | OCTREE_EXTRACT_Z(i);
            }
        }
    } else {
        //std::cout << "Recursing" << std::endl;
        auto node = this->container->getNode(layer, index);
        for(int i = 0; i < OCTREE_SIZE; i++) {
            int child = node->getChildByIdx(i);
            if(child != NO_NODE) {
                //std::cout << "Child in idx " << i << " exists" << std::endl;
                this->buildChildList(
                    layersToGo-1, 
                    childList, 
                    childCount, 
                    layer+1, 
                    child, 
                    (x << 1) | OCTREE_EXTRACT_X(i),
                    (y << 1) | OCTREE_EXTRACT_Y(i),
                    (z << 1) | OCTREE_EXTRACT_Z(i));
            }
        }
    }
}

void DctEncoder::workerInner() {
    int job = this->getJob();
    while(job != -1) {
        std::cout << "Starting job " << job << std::endl;
        /*
        std::cout << "Looking for defects" << std::endl;
        for(int i = 0; i < OCTREE_MAX_DEPTH; i++) {
            for(int j = 0; j < this->container->getLayerSize(i); j++) {
                auto node = this->container->getNode(i, j);
                for(int k = 0; k < OCTREE_SIZE; k++) {
                    auto child = node->getChildByIdx(k);
                    if(child < 0 && child != NO_NODE) {
                        throw std::runtime_error("Invalid structure: Child of " + 
                            std::to_string(i) + 
                            ", " + 
                            std::to_string(j) + 
                            ": Child " + 
                            std::to_string(k) + 
                            " is " + 
                            std::to_string(child)
                        );
                    }
                }
            }
        }*/
        for(int i = job; i < std::min(this->layerSize, job + this->jobSize); i++) {
            DctChildEntry childList[8*8*8];
            int childCount = 0;

            int* table_position = &this->layerPtr[i* DCT_TABLE_SIZE];

            float color_table[DCT_TABLE_SIZE];
            // Fill the table with averages
            // TODO smarter than this?
            float avg = container->getNode(layer, i)->getPayload()->yuv.x;
            for(int j = 0; j < DCT_TABLE_SIZE; j++) {
                color_table[j] = avg;
            }
            // Put in actual colors
            //std::cout << "Building children...." << std::endl;
            this->buildChildList(2, childList, &childCount, layer, i, 0, 0, 0);
            //std::cout << childCount << " nodes" << std::endl;
            //std::cout << "Layer: " << layer << std::endl;
            for(int c = 0; c < childCount; c++) {
                auto child = childList[c];
                int addr = DCT_ADDRESS(child.x, child.y, child.z);

                auto node = container->getNode(layer+3, child.index);
                //std::cout << "Layer " << (layer + 3) << std::endl;
                auto yuv = node->getPayload()->yuv;
                float y = yuv[this->colorIdx];
                //std::cout << "Coordinate: " << child.x << " " << child.y << " " << child.z << std::endl;
                //std::cout << "YUV: " << yuv.x << " " << yuv.y << " " << yuv.z << std::endl;
                auto rgb = node->getPayload()->data;
                //std::cout << "RGB: " << rgb.x << " " << rgb.y << " " << rgb.z << std::endl;
                color_table[addr] = y;
            }
            // run DCT
            float dct_table[DCT_TABLE_SIZE];
            do_dct(color_table, dct_table);

            // Quantize
            //int quantization_table[DCT_SIZE * DCT_SIZE * DCT_SIZE];
            do_quantization(dct_table, table_position);
        }
        std::cout << "Done with " << job << std::endl;
        job = this->getJob();
    }
    std::cout << "Work thread done" << std::endl;
}

int DctEncoder::getJob() {
    std::lock_guard<std::mutex>(this->jobMutex);
    if(this->jobs.empty()) {
        return -1;
    }
    int job = this->jobs.front();
    this->jobs.pop();
    return job;
}

void DctEncoder::run(int paralellism) {
    std::vector<std::thread*> threadPool;
    for(int i = 0; i < paralellism; i++) { 
        threadPool.push_back(new std::thread(DctEncoder::worker, this));
    }
    for(auto thread : threadPool) {
        thread->join();
        delete thread;
    }
}