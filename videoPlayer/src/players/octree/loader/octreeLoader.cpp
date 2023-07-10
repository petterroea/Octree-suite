#include "octreeLoader.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

#include <zlib.h>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

#include <dct2d/quantization.h>
#include <dct2d/dct.h>
#include <dct2d/yuv.h>

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
    this->ioMutex.lock();
    std::cout << "Frameset requested: from " << frameset->getStartIndex() << " to " << frameset->getEndIndex() << std::endl;
    if(this->requestedFrameset != nullptr) {
        std::cout << "A frame is already requested, queueing this for next" << std::endl;
        this->nextRequestedFrameset = frameset;
    } else {
        std::cout << "Frame coming right up" << std::endl;
        this->requestedFrameset = frameset;
    }
    if(!this->isLoadingThreadRunning()) {
        std::cout << "Loading thread not running, starting it." << std::endl;
        this->startLoadingThread();
    } else {
        std::cout << "Loading thread is already running, no need to start" << std::endl;
    }
    std::cout << "Done handling frame request" << std::endl;
    this->ioMutex.unlock();
}

bool OctreeLoader::isLoadingThreadRunning() { 
    return this->threadRunning;
}

void OctreeLoader::startLoadingThread() {
    if(this->isLoadingThreadRunning()) { 
        throw std::runtime_error("Tried to start the loading thread twice!");
    }
    std::cout << "Starting loading thread ==============================" << std::endl;
    // If the loading thread already exists it halted, delete it before restarting
    if(this->loadingThread) {
        std::cout << "A loadingThread thread already exists, we have to re-start it." << std::endl;
        this->loadingThread->join();
        delete this->loadingThread;
    }
    this->loadingThread = new std::thread(OctreeLoader::worker, this);
    this->threadRunning = true;
}

void OctreeLoader::calculateLeafFlags(int layer, int index, LayeredOctreeContainerStatic<octreeColorType>* container) {
    bool has_children = false;
    auto node = container->getNode(layer, index);

    uint8_t leaf_flags = 0;

    for(int i = 0; i < OCTREE_SIZE; i++) {
        auto child = node->getChildByIdx(i);
        if(child != NO_NODE) {
            if(!container->getNode(layer+1, child)->getChildFlags()) {
                leaf_flags |= (1 << i);
            }
        }
    }

    node->setLeafFlags(leaf_flags);
}

void OctreeLoader::worker(OctreeLoader* me) {
    // Check if there is a frame to load. Should always be true at this point,
    // since the thread only runs if there is a frame to load.
    me->ioMutex.lock();
    std::cout << "Loading thread started" << std::endl;
    bool hasNextFrameset = me->requestedFrameset != nullptr;

    while(hasNextFrameset) {
        me->ioMutex.unlock();
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

        //Load quantization settings
        unsigned char quantization_start, quantization_end;
        file.read(reinterpret_cast<char*>(&quantization_start), sizeof(quantization_start));
        file.read(reinterpret_cast<char*>(&quantization_end), sizeof(quantization_start));

        std::cout << "Read quantization settings: start " << quantization_start << " end " << quantization_end << std::endl;

        build_quantization_lookup_table(quantization_start, quantization_end);

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
            me->loadLayer(i, layerSizes, container, file);
        }

        file.close();
        // Set leaf flags
        for(int i = 0; i < layerSizes[0]; i++) {
            me->calculateLeafFlags(0, i, container);
        }

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
        // Do not unlock the mutex until we have more work or the thread is dead
    }
    me->threadRunning = false;
    std::cout << "!!!!!!! Loading thread finished" << std::endl;
    // Make sure 
    me->ioMutex.unlock();
}
void OctreeLoader::loadLayer(int layer, std::vector<int>& layerSizes, LayeredOctreeContainerStatic<octreeColorType>* container, std::ifstream& file) {
    unsigned char layerCookie;
    file.read(reinterpret_cast<char*>(&layerCookie), sizeof(char));
    if(layerCookie != 0x13) {
        throw std::runtime_error("Invalid layer cookie");
    }

    int layerSize = layerSizes[layer];
    std::cout << "Layer " << layer << " size " << layerSize << std::endl;
    if(layerSize == 0) {
        return;
    }

    auto layerPtr = container->getNode(layer, 0);
    if(!layerPtr) {
        throw std::runtime_error("Failed to get layerPtr for layer " + std::to_string(layer));
    }
    // Read the data
    int readLength;

    unsigned char* y_data = this->readCompressedImage(&readLength, file);
    unsigned char* u_data = this->readCompressedImage(&readLength, file);
    unsigned char* v_data = this->readCompressedImage(&readLength, file);

    unsigned char* child_flag_data = this->readCompressed(&readLength, file);

    int* childPointers = nullptr;
    int childPtrIdx = 0;

    int child_ptr_count = 0;
    for(int i = 0; i < layerSize; ++i) {
        child_ptr_count += std::popcount(child_flag_data[i]);
    }
    if(child_ptr_count != 0) {
        childPointers = reinterpret_cast<int*>(this->readCompressed(&readLength, file));
    }
    int lastChild = 0;

    //For each node in the layer, read it and unpack it
    for(int n = 0; n < layerSize; n++) {
        auto rgb = yuv_to_rgb(glm::vec3(
            y_data[n],
            u_data[n],
            v_data[n]
        ));

        uint8_t childFlags = child_flag_data[n];

        *layerPtr = LayeredOctree<octreeColorType>(glm::vec3(
            static_cast<float>(rgb.r)/255.0f, 
            static_cast<float>(rgb.g)/255.0f, 
            static_cast<float>(rgb.b)/255.0f
        ));

        //layerPtr->setLeafFlags(leafFlags);

        // Load children
        for(int c = 0; c < OCTREE_SIZE; c++) {
            if((childFlags >> c) & 1) {
                layer_ptr_type childPtr = childPointers[childPtrIdx++];
                childPtr += lastChild;
                lastChild = childPtr;
                //file.read(reinterpret_cast<char*>(&childPtr), sizeof(layer_ptr_type));
                if(childPtr > layerSizes[layer+1]) {
                    throw std::runtime_error("Invalid octree: node " + 
                        std::to_string(n) + 
                        " has child ptr " + 
                        std::to_string(childPtr) + 
                        ", which is larger than " + 
                        std::to_string(layerSizes[layer+1])
                    );
                }

                layerPtr->setChild(childPtr, c, 0);
            }
        }
        layerPtr++;
    }
    // Read the layer
    //file.read((char*)layer, layerSize * sizeof(LayeredOctree<octreeColorType>));
    delete[] y_data;
    delete[] u_data;
    delete[] v_data;

    delete[] child_flag_data;
    //delete[] leaf_flag_data;

    if(childPointers) {
        delete[] childPointers;
    }
}

OctreeFrameset* OctreeLoader::peekLoadedOctreeFrameset() {
    return this->loadedFrameset;
}
// Returns the currently loaded octree and gives away ownership of it
loadedOctreeType* OctreeLoader::getLoadedOctree(OctreeFrameset** frameset) {
    std::lock_guard mutex(this->ioMutex);
    auto val = this->loadedOctree;
    if(this->loadedOctree != nullptr) {
        *frameset = this->loadedFrameset;
    }
    this->loadedOctree = nullptr;

    return val;
}


OctreeFrameset* OctreeLoader::getCurrentlyLoadingFrameset() {
    return this->requestedFrameset;
}

OctreeFrameset* OctreeLoader::getNextLoadingFrameset() {
    return this->nextRequestedFrameset;
}

unsigned char* OctreeLoader::readCompressedImage(int* bufferLength, std::ifstream& file) {
    int length, dctUnits;
    file.read(reinterpret_cast<char*>(&length), sizeof(length));
    file.read(reinterpret_cast<char*>(&dctUnits), sizeof(dctUnits));

    std::cout << "Reading compressed image length " << length << " dct units " << dctUnits << std::endl;

    unsigned char* data = new unsigned char[length];

    //Read and decompress DCT
    int read_size;
    if(dctUnits > 0) {
        int* quantizationArray = reinterpret_cast<int*>(this->readCompressed(&read_size, file));

        float dctScratchpad[DCT_SIZE * DCT_SIZE];
        for(int i = 0; i < dctUnits; i++) {
            unsigned char* currentPtr = &data[i * DCT_SIZE * DCT_SIZE];
            do_de_quantization(&quantizationArray[i * DCT_SIZE * DCT_SIZE], dctScratchpad);
            reverse_dct(dctScratchpad, currentPtr);
        }
        delete[] quantizationArray;
    }

    // Read the last data
    int remainingData = length - dctUnits * DCT_SIZE * DCT_SIZE;
    std::cout << "Remaining " << remainingData << std::endl;
    if(remainingData > 0) {
        file.read(reinterpret_cast<char*>(&data[dctUnits*DCT_SIZE*DCT_SIZE]), remainingData);
    }
    return data;
}

unsigned char* OctreeLoader::readCompressed(int* bufferSize, std::ifstream& file) {
    std::cout << "Reading compressed" << std::endl;
    unsigned char cookie;

    file.read(reinterpret_cast<char*>(&cookie), sizeof(cookie));
    if(cookie != 0x69) {
        throw std::runtime_error("Invalid cookie: " + std::to_string(cookie));
    }

    file.read(reinterpret_cast<char*> (bufferSize), sizeof(int));
    unsigned long originalSize = *bufferSize;

    int compressedSize = 0;
    file.read(reinterpret_cast<char*> (&compressedSize), sizeof(compressedSize));
    unsigned char* buffer = new unsigned char[*bufferSize];
    
    unsigned char* compressedBuffer = new unsigned char[compressedSize];
    file.read(reinterpret_cast<char*>(compressedBuffer), compressedSize);

    int z_result = uncompress(buffer, &originalSize, compressedBuffer, compressedSize);

    if(z_result != Z_OK) {
        if(z_result == Z_MEM_ERROR ) {
            throw std::runtime_error("Zlib out of memory");
        } else if(z_result == Z_BUF_ERROR) {
            throw std::runtime_error("Zlib error: too small output buffer: " + std::to_string(originalSize));
        }
        throw std::runtime_error("Failed to zlib compress: " + std::to_string(z_result));
    }
    file.read(reinterpret_cast<char*>(&cookie), sizeof(cookie));
    if(cookie != 0x68) {
        throw std::runtime_error("Invalid post-cookie: " + std::to_string(cookie));
    }

    std::cout << "Read " << compressedSize << " bytes, inflated to " << originalSize << std::endl;
    *bufferSize = originalSize;

    return buffer;
}