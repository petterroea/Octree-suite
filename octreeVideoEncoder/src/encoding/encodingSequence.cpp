#include "encodingSequence.h"

#include <math.h>
#include <iostream>
#include <fstream>
#include <exception>
#include <thread>

#include <octree/pointerOctree.h>

#include <glm/vec3.hpp>

#include "../config.h"

EncodingSequence::EncodingSequence(OctreeSequence* sequence, int from, int to, std::string fullPath, VideoEncoderRunArgs* args): 
    fullPath(fullPath),
    sequence(sequence), 
    from(from), 
    to(to),
    args(args) {
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
        //std::cout << "Installed octree with rootidx " << rootIdx << std::endl;

        this->populateHashmap(0, rootIdx, layeredContainer, 8);
    }

    // TODO improve
    // We do not deduplicate the first layers
    for(int i = 3; i < OCTREE_MAX_DEPTH; i++) {
        std::cout << "------------------------------Deduplicating " << i << std::endl;
        this->deduplicator = new DeDuplicator(this->hashmaps[i], layeredContainer, i, this->args);
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
    if(max_depth == depth) {
        return;
    }
    auto tree = octreeContainer.getNode(depth, idx);
    this->hashmaps[depth].push(tree->getHashKey(), idx);
    for(int i = 0; i < 8; i++) {
        auto child = tree->getChildByIdx(i);
        if(child != NO_NODE) {
            this->populateHashmap(depth+1, child, octreeContainer, max_depth);
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
    int nodeCount[OCTREE_MAX_DEPTH];
    int childPtrCount[OCTREE_MAX_DEPTH];

    int totalPreTrimCount = 0;
    int postTrimCount = 0;
    for(int i = 0; i < OCTREE_MAX_DEPTH; i++) {
        totalPreTrimCount += trees.getLayerSize(i);
        int count = 0;
        int childCount = 0;
        for(int x = 0; x < trees.getLayerSize(i); x++) {
            auto node = trees.getNode(i, x);
            if(!node->getPayload()->trimmed) {
                postTrimCount++;
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
        totalSize += nodeSize*nodeCount[i] + sizeof(layer_ptr_type) * childPtrCount[i];
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
        //file.write(reinterpret_cast<char*>(&currentLayerOffset), sizeof(int));
        int layerSize = nodeCount[i];
        file.write(reinterpret_cast<char*>(&layerSize), sizeof(int));
        currentLayerOffset += nodeSize*nodeCount[i] + childPtrCount[i] * sizeof(layer_ptr_type);
    }

    // Write the payload
    // By writing the bottom layers first, we can write the file in one go, 
    // even with having to take into account that child pointers change due to node trimming
    for(int i = OCTREE_MAX_DEPTH-1; i >= 0 ; i--) {
        std::cout << "Writing layer ( " << trees.getLayerSize(i) << " nodes, " << nodeCount[i] << " after trim)" << i << std::endl;
        long layerDistCount = 0;
        int measurementCounts = 0;

        int measureChildCount = 0;
        int childMeasurements = 0;

        int writtenNodes = 0;
        for(int x = 0; x < trees.getLayerSize(i); x++) {
            auto node = trees.getNode(i, x);
            auto payload = node->getPayload();
            if(!payload->trimmed) {
                payload->writtenOffset = writtenNodes++;
                //std::cout << "Node " << x << ": NOT TRIMMED, written at " << payload->writtenOffset << std::endl;
                // Write the payload, converting glm::vec3 to rgb bytes
                char cookie = 0x69;
                file.write(&cookie, sizeof(cookie));

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

                int index_offset = -1;

                // write child pointers
                int children = 0;
                for(int j = 0; j < OCTREE_SIZE; j++) {
                    auto child_idx = node->getChildByIdx(j);
                    if(child_idx != NO_NODE) {
                        // Figure out if the node was trimmed
                        if(i != OCTREE_MAX_DEPTH - 1) { // Not needed?
                            auto child_node = trees.getNode(i+1, child_idx);
                            auto payload = child_node->getPayload();
                            if(payload->trimmed) {
                                // The node is trimmed, figure out the index of the node that replaced it
                                // This is a two-step operation:
                                // 1. Find the index of the node that replaces the trimmed child
                                // 2. Find the actual written index of the replacing node
                                if(payload->replacement == NO_NODE) {
                                    throw std::runtime_error("Layer " + 
                                        std::to_string(i) + 
                                        " node " + 
                                        std::to_string(x) + 
                                        ": child " + 
                                        std::to_string(j) +
                                        " ( node " +
                                        std::to_string(child_idx) +
                                        " ) is trimmed, but no replacement");
                                }
                                auto replacement_node = trees.getNode(i+1, payload->replacement);
                                auto replacement_payload = replacement_node->getPayload();
                                if(replacement_payload->trimmed) {
                                    throw std::runtime_error(
                                        "Node tried to be replaced with a node that is trimmed: " +
                                        std::to_string(child_idx) +
                                        " was replaced with " +
                                        std::to_string(payload->replacement) +
                                        ", which is trimmed ( replaced with " +
                                        std::to_string(replacement_payload->replacement) +
                                        " )."
                                    );
                                }
                                if(replacement_payload->writtenOffset == -1) {
                                    throw std::runtime_error("Tried to determine a child index to a node that isn't written");
                                }
                                //std::cout << " Child " << j << " ( " << child_idx << " ) is trimmed. Replacement: " << replacement_payload->writtenOffset << std::endl;
                                child_idx = replacement_payload->writtenOffset;
                                //std::cout << "    New child index: " << child_idx << std::endl;
                            } else {
                                //std::cout << " Child " << j << " Not trimmed " << std::endl;
                                child_idx = payload->writtenOffset;
                                if(child_idx == NO_NODE) {
                                    throw std::runtime_error("got child that hasn't been written yet");
                                }
                            }
                        }
                        if(child_idx == NO_NODE) {
                            throw std::runtime_error("Trying to write negative index, bad!");
                        }
                        // Stats measurement
                        if(index_offset == -1) {
                            index_offset = child_idx;
                        } else {
                            int dist = child_idx - index_offset;
                            //std::cout << "Dist: " << dist << std::endl;
                            if(dist > 128) {
                                //throw std::runtime_error("didnt expect that");
                            }
                            layerDistCount += dist;
                            measurementCounts++;
                        }
                        children++;
                        file.write(reinterpret_cast<char*>(&child_idx), sizeof(layer_ptr_type));
                    }
                }
                measureChildCount += children;
                childMeasurements++;
            } else {
                //std::cout << "Node " << x << ": VERY TRIMMED" << std::endl;
            }
        }
        // Verify that we didn't write more nodes than we calculated
        if(writtenNodes > nodeCount[i]) {
            throw std::runtime_error("Wrote more nodes than expected...");
        }
        if(measurementCounts != 0) {
            std::cout << "Average distance between nodes: " << (layerDistCount / measurementCounts) << std::endl;
        }
        if(childMeasurements != 0) {
            std::cout << "Average child count: " << (measureChildCount / childMeasurements) << "( total " << measureChildCount << " )" << std::endl;
        }
    }

    std::cout << "Done, wrote to " << filename << std::endl;
    int nodesTrimmed = totalPreTrimCount - postTrimCount;
    std::cout << "Trim stats: " << 
        nodesTrimmed << 
        " trimmed out of " << 
        totalPreTrimCount << 
        " nodes. ( " <<
        ( static_cast<float>(nodesTrimmed) / static_cast<float>(totalPreTrimCount) * 100.0f ) << 
        " % )" << 
        std::endl;
    file.close();
}