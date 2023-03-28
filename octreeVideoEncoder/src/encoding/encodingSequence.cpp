#include "encodingSequence.h"

#include <math.h>
#include <iostream>
#include <fstream>
#include <exception>
#include <thread>

#include <octree/pointerOctree.h>

#include <glm/vec3.hpp>

#include "../config.h"

EncodingSequence::EncodingSequence(OctreeSequence* sequence, int from, int to, std::string fullPath): 
    fullPath(fullPath),
    sequence(sequence), 
    from(from), 
    to(to) {
    if(from < 0 || to > sequence->getFrameCount()) {
        throw std::invalid_argument("Invalid frame range");
    }
}

void EncodingSequence::encode() {
    PointerOctree<octreeColorType>** octrees = new PointerOctree<octreeColorType>*[this->to - this->from + 1];
    for(int frame = this->from; frame <= this->to; frame++) {
        std::cout << "Loading frame " << frame << std::endl;
        octrees[frame - this->from] = this->sequence->getOctree(frame);
        std::cout << "Octree " << frame << " fill rate: " << pointerOctreeFillRate(octrees[frame - this->from]) << std::endl;
    }
    std::cout << "Calculating deltas" << std::endl;
    for(int frame = this->from; frame <= this->to - 1; frame++) {
        auto lhs = octrees[frame - this->from];
        auto rhs = octrees[frame - this->from + 1];
        std::cout << "Color diff " << frame << " to " << (frame+1) << ": " << diffPointerOctreeColor(lhs, rhs) << std::endl;
        std::cout << "Octree diff " << frame << " to " << (frame+1) << ": " << pointerOctreeSimilarity(lhs, rhs) << std::endl;
    }
    // Make the trees exist in the same context, then put them in a hashmap
    LayeredOctreeProcessingContainer<octreeColorType> layeredContainer;

    for(int frame = this->from; frame <= this->to; frame++) {
        auto tree = octrees[frame - this->from];
        int rootIdx = layeredContainer.addOctree(tree);
        std::cout << "Installed octree with rootidx " << rootIdx << std::endl;

        this->populateHashmap(0, rootIdx, layeredContainer, 8);
    }

    // TODO improve
    // We do not deduplicate the first layers
    for(int i = 3; i < OCTREE_MAX_DEPTH; i++) {
        std::cout << "------------------------------Deduplicating " << i << std::endl;
        this->deduplicator = new DeDuplicator(this->hashmaps[i], layeredContainer, i, std::thread::hardware_concurrency());
        this->deduplicator->run();
        delete this->deduplicator; //TODO
    }

    // Write the tree to disk
    this->writeToDisk(layeredContainer, this->fullPath);

    // Re-assemble trees
    std::cout << "Cleaning up" << std::endl;
    for(int frame = this->from; frame <= this->to; frame++) {
        delete octrees[frame - this->from];
    }
    delete[] octrees;
}

void EncodingSequence::populateHashmap(int depth, int idx, LayeredOctreeProcessingContainer<octreeColorType>& octreeContainer, int max_depth) {
    if(!max_depth) {
        return;
    }
    auto tree = octreeContainer.getNode(depth, idx);
    this->hashmaps[depth].push(tree->getHashKey(), idx);
    for(int i = 0; i < 8; i++) {
        auto child = tree->getChildByIdx(i);
        if(child != NO_NODE) {
            this->populateHashmap(depth+1, child, octreeContainer, max_depth-1);
        }
    }
}
void EncodingSequence::writeToDisk(LayeredOctreeProcessingContainer<octreeColorType>& trees, std::string filename) {
    //Set up vectors for storing the output nodes
    /*
    std::vector<LayeredOctree> outputLayers[OCTREE_MAX_DEPTH];
    for(int i = 0; i < OCTREE_MAX_DEPTH; i++) {
        outputLayers[i] = std::vector<LayeredOctree>();
    }
    */
    int bytesWritten[OCTREE_MAX_DEPTH];
    int nodeCount[OCTREE_MAX_DEPTH];
    int childPtrCount[OCTREE_MAX_DEPTH];
    for(int i = 0; i < OCTREE_MAX_DEPTH; i++) {
        bytesWritten[i] = 0;
        int count = 0;
        int childCount = 0;
        for(int x = 0; x < trees.getLayerSize(i); x++) {
            auto node = trees.getNode(i, x);
            if(!node->getPayload()->trimmed) {
                count++;
                childCount += node->getChildCount();
            }
        }
        nodeCount[i] = count;
        childPtrCount[i] = childCount;
    }

    /* 
     * Calculate a pessimistic node size and use it
     * Size of payload + all flags. Child pointer size encoded using childPtrCount
    */
    // Colors take 3 bytes
    int nodeSize = sizeof(uint8_t) * 3 + 3 * sizeof(uint8_t);

    int fileHeaderSize = sizeof(int)*3 + OCTREE_MAX_DEPTH * sizeof(int);

    int totalSize = fileHeaderSize;
    for(int i = 0; i < OCTREE_MAX_DEPTH; i++) {
        totalSize += nodeSize*nodeCount[i] + sizeof(int) * childPtrCount[i];
    }
    std::cout << "Expecting an output file around " << totalSize << " (" << (totalSize / 1024 / 1024) << " mb)" << std::endl;

    // Write the file header
    std::ofstream file;
    file.open(filename, std::ios::out | std::ios::binary);

    if(!file.is_open()) {
        throw std::invalid_argument("Could not open file for writing");
    }

    int magic = 0xfade1337;
    int max_tree_depth = OCTREE_MAX_DEPTH;

    // Write the tree header
    file.write(reinterpret_cast<char*>(&magic), sizeof(int));
    file.write(reinterpret_cast<char*>(&max_tree_depth), sizeof(int));
    file.write(reinterpret_cast<char*>(&fileHeaderSize), sizeof(int));

    int currentLayerOffset = 0;
    for(int i = 0; i < OCTREE_MAX_DEPTH; i++) {
        file.write(reinterpret_cast<char*>(&currentLayerOffset), sizeof(int));
        currentLayerOffset += nodeSize*nodeCount[i] + childPtrCount[i] * sizeof(layer_ptr_type);
    }

    // Write the payload
    for(int i = 0; i < OCTREE_MAX_DEPTH; i++) {
        for(int x = 0; x < trees.getLayerSize(i); x++) {
            auto node = trees.getNode(i, x);
            if(!node->getPayload()->trimmed) {
                // Write the payload, converting glm::vec3 to rgb bytes
                auto payload = node->getPayload();
                uint8_t r = static_cast<char>(payload->data.x*256.0f);
                uint8_t g = static_cast<char>(payload->data.y*256.0f);
                uint8_t b = static_cast<char>(payload->data.z*256.0f);
                file.write(reinterpret_cast<char*>(&r), sizeof(uint8_t));
                file.write(reinterpret_cast<char*>(&g), sizeof(uint8_t));
                file.write(reinterpret_cast<char*>(&b), sizeof(uint8_t));

                uint8_t childCount = node->getChildCount();
                uint8_t childFlags = node->getChildFlags();
                uint8_t leafFlags = node->getLeafFlags();

                file.write(reinterpret_cast<char*>(&childCount), sizeof(uint8_t));
                file.write(reinterpret_cast<char*>(&childFlags), sizeof(uint8_t));
                file.write(reinterpret_cast<char*>(&leafFlags), sizeof(uint8_t));

                // write child pointers
                for(int j = 0; j < OCTREE_SIZE; j++) {
                    auto child_idx = node->getChildByIdx(j);
                    if(child_idx != NO_NODE) {
                        file.write(reinterpret_cast<char*>(&child_idx), sizeof(layer_ptr_type));
                    }
                }
            }
        }
    }

    std::cout << "Done, wrote to " << filename << std::endl;
    file.close();
}