#include "octreeLoader.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

OctreeLoader::OctreeLoader(LayeredOctreeAllocator* allocator): 
    lastLoadedFrameset(nullptr),
    octreeContainer(nullptr) {
    this->allocator = allocator;
}

OctreeLoader::~OctreeLoader() {
    std::cout << "Shutting down OctreeLoader" << std::endl;
    if(this->loadingThread != nullptr) {
        this->loadingThread->join();
    }
    delete this->loadingThread;
}

void OctreeLoader::requestFrameset(OctreeFrameset* frameset) {
    std::cout << "Frameset requested" << std::endl;
    std::cout << ": From " << frameset->getStartIndex() << " to " << frameset->getEndIndex() << std::endl;
    if(this->requestedFrameset != nullptr) {
        std::cout << "A frame is already requested, queueing this for next" << std::endl;
        this->nextRequestedFrameset = frameset;
    } else {
        std::cout << "Frame coming right up" << std::endl;
        this->requestedFrameset = frameset;
    }
    if(!this->isLoadingThreadRunning()) {
        std::cout << "Starting loading thread" << std::endl;
        this->startLoadingThread();
    }
    std::cout << "Starting loading thread" << std::endl;
}

bool OctreeLoader::isLoadingThreadRunning() { 
    // TODO wrong
    // Sorry future me!
    return this->loadingThread != nullptr;
}

void OctreeLoader::startLoadingThread() {
    if(this->isLoadingThreadRunning()) { 
        throw std::runtime_error("Tried to start the loading thread twice!");
    }
    std::cout << "Starting loading ==============================" << std::endl;
    this->loadingThread = new std::thread(OctreeLoader::worker, this);
}

void OctreeLoader::worker(OctreeLoader* me) {
    std::cout << "Loading thread started" << std::endl;

    // Check if there is a frame to load. Should always be true at this point,
    // since the thread only runs if there is a frame to load.
    me->ioMutex.lock();
    bool hasNextFrameset = me->requestedFrameset != nullptr;
    me->ioMutex.unlock();

    while(hasNextFrameset) {
        std::cout << "-> Loading a frame" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        std::cout << "From " << me->requestedFrameset->getStartIndex() << " to " << me->requestedFrameset->getEndIndex() << std::endl;

        // Open the file
        std::string fileName = me->requestedFrameset->getFilename();
        std::ifstream file(fileName.c_str(), std::ios::binary);
        if(!file.is_open()) {
            std::cout << "fuck" << std::endl;
            std::cout << "Could not open file " << fileName << std::endl;
            throw std::runtime_error("Could not open file " + fileName);
        }
        // Read the magic, max tree depth, and header size to integers
        int magic;
        file.read((char*)&magic, sizeof(int));
        int maxDepth;
        file.read((char*)&maxDepth, sizeof(int));
        int headerSize;
        file.read((char*)&headerSize, sizeof(int));

        if(magic != 0xfade1337) {
            std::cout << "Invalid magic number in file " << fileName << std::endl;
            throw std::runtime_error("Invalid magic number in file " + fileName);
        }

        // Read the layer sizes
        std::vector<int> layerSizes;
        layerSizes.resize(maxDepth);
        file.read((char*)&layerSizes[0], maxDepth * sizeof(int));

        std::cout << "Read file header, layers: " << std::endl;
        for(auto layer : layerSizes) {
            std::cout << layer << std::endl;
        }

        // Allocate a layered octree
        LayeredOctreeContainerStatic<octreeColorType>* container = me->allocator->allocate(layerSizes);

        // Load each layer
        for(int i = layerSizes.size()-1; i >= 0; i--) {
            std::cout << "Layer " << i << " size " << layerSizes[i] << std::endl;
            int layerSize = layerSizes[i];

            auto layerPtr = container->getNode(i, 0);
            if(!layerPtr) {
                throw std::runtime_error("Failed to get layerPtr for layer " + std::to_string(i));
            }
            //For each node in the layer, read it and unpack it
            for(int n = 0; n < layerSizes[i]; n++) {
                uint8_t r, g, b;

                uint8_t childCount, childFlags, leafFlags;

                char cookie;
                file.read(reinterpret_cast<char*>(&cookie), sizeof(char));
                if(cookie != 0x69) {
                    throw std::runtime_error("Cookie was incorrect, we are not aligned");
                }

                file.read(reinterpret_cast<char*>(&r), sizeof(uint8_t));
                file.read(reinterpret_cast<char*>(&g), sizeof(uint8_t));
                file.read(reinterpret_cast<char*>(&b), sizeof(uint8_t));

                file.read(reinterpret_cast<char*>(&childCount), sizeof(uint8_t));
                file.read(reinterpret_cast<char*>(&childFlags), sizeof(uint8_t));
                file.read(reinterpret_cast<char*>(&leafFlags), sizeof(uint8_t));

                *layerPtr = LayeredOctree<octreeColorType>(glm::vec3(
                    static_cast<float>(r)/255.0f, 
                    static_cast<float>(g)/255.0f, 
                    static_cast<float>(b)/255.0f
                ));

                layerPtr->setChildFlags(childFlags);
                layerPtr->setLeafFlags(leafFlags);

                assert(childCount <= 8);

                // Load children
                for(int c = 0; c < OCTREE_SIZE; c++) {
                    if((childFlags >> c) & 1) {
                        layer_ptr_type childPtr;
                        file.read(reinterpret_cast<char*>(&childPtr), sizeof(layer_ptr_type));
                        if(childPtr > layerSizes[i+1]) {
                            throw std::runtime_error("Invalid octree: node " + 
                                std::to_string(n) + 
                                " has child ptr " + 
                                std::to_string(childPtr) + 
                                ", which is larger than " + 
                                std::to_string(layerSizes[i+1])
                            );
                        }

                        layerPtr->setChild(childPtr, c, (leafFlags >> c & 1));
                    }
                }
                layerPtr++;
            }
            // Read the layer
            //file.read((char*)layer, layerSize * sizeof(LayeredOctree<octreeColorType>));
        }

        file.close();

        // Upload to CUDA
        // TODO memory manage this shit
        LayeredOctreeContainerCuda<octreeColorType>* cudaContainer = new LayeredOctreeContainerCuda<octreeColorType>(*container);
        me->allocator->free(container);

        auto end = std::chrono::high_resolution_clock::now();

        auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Done reading " << fileName << " in " << load_time.count() << " ms." << std::endl;

        // Prepare for loading next frame
        me->ioMutex.lock();
        // Put this frame in the "Done loading" slot
        if(me->loadedOctree != nullptr) {
            std::cout << "Loaded a frame, but the last one isnt consumed yet?" << std::endl;
            delete me->loadedOctree;
        }
        me->loadedOctree = cudaContainer;
        me->loadedFrameset = me->requestedFrameset;

        me->requestedFrameset = me->nextRequestedFrameset;
        me->nextRequestedFrameset = nullptr;
        hasNextFrameset = me->requestedFrameset != nullptr;

        me->ioMutex.unlock();
    }
    std::cout << "!!!!!!! Loading thread finished" << std::endl;
}

// Returns the currently loaded octree and gives away ownership of it
loadedOctreeType* OctreeLoader::getLoadedOctree(OctreeFrameset** frameset) {
    std::lock_guard mutex(this->ioMutex);
    auto val = this->loadedOctree;
    this->loadedOctree = nullptr;

    *frameset = this->loadedFrameset;
    return val;
}


OctreeFrameset* OctreeLoader::getCurrentlyLoadingFrameset() {
    return this->requestedFrameset;
}

OctreeFrameset* OctreeLoader::getNextLoadingFrameset() {
    return this->nextRequestedFrameset;
}