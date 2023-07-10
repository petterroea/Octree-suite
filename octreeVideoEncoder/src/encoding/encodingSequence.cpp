#include "encodingSequence.h"

#include <math.h>
#include <iostream>
#include <fstream>
#include <exception>
#include <thread>

#include <zlib.h>

#include <octree/pointerOctree.h>

#include <glm/vec3.hpp>

#include <dct2d/yuv.h>
#include <dct2d/dct.h>
#include <dct2d/quantization.h>

#include "../config.h"

void EncodingSequence::write_color_compressed(unsigned char* colors, int length, std::ofstream& file, std::ofstream* stats_file) {
    int dctUnits = length / (DCT_SIZE*DCT_SIZE);
    file.write(reinterpret_cast<const char*>(&length), sizeof(length));
    file.write(reinterpret_cast<const char*>(&dctUnits), sizeof(dctUnits));
    if(stats_file) {
        *stats_file << "color->,";
    }

    std::cout << "Writing color compress of size " << dctUnits << " units" << std::endl;

    if(dctUnits != 0) {
        int* quantizationArray = new int[dctUnits*DCT_SIZE*DCT_SIZE];
        //DCT encode and write the units
        float dctArray[DCT_SIZE*DCT_SIZE];
        for(int i = 0; i < dctUnits; i++) {
            unsigned char* data = &colors[i*DCT_SIZE*DCT_SIZE];
            do_dct(data, dctArray);
            do_quantization(dctArray, &quantizationArray[i*DCT_SIZE*DCT_SIZE]);
        }
        this->write_compressed((unsigned char*)quantizationArray, dctUnits*DCT_SIZE*DCT_SIZE*sizeof(int), file, stats_file);
        delete[] quantizationArray;
    }
    if(stats_file) {
        *stats_file << "<-,";
    }

    // Write remaining data that couldn't be DCT compressed
    int remaining = length - dctUnits * DCT_SIZE * DCT_SIZE;
    if(remaining != 0) {
        file.write(reinterpret_cast<const char*>(&colors[dctUnits * 8]), remaining);
    }
}

int EncodingSequence::write_compressed(unsigned char* data, int length, std::ofstream& file, std::ofstream* stats_file) {
    //Write a test cookie
    unsigned char cookie = 0x69;
    file.write(reinterpret_cast<const char*>(&cookie), sizeof(cookie));

    unsigned long outBufferSize = static_cast<unsigned long>(static_cast<float>(length)*1.1f + 21);
    unsigned char* compressed = new unsigned char[outBufferSize];
    unsigned long written_bytes = outBufferSize;
    int z_result = compress(compressed, &written_bytes, data, length);
    if(z_result != Z_OK) {
        if(z_result == Z_MEM_ERROR ) {
            throw std::runtime_error("Zlib out of memory");
        } else if(z_result == Z_BUF_ERROR) {
            throw std::runtime_error("Zlib error: too small output buffer: " + std::to_string(outBufferSize));
        }
        throw std::runtime_error("Failed to zlib compress: " + std::to_string(z_result));
    }

    // Write original length
    file.write(reinterpret_cast<const char*>(&length), sizeof(int));
    // Write the encoded length
    file.write(reinterpret_cast<const char*>(&written_bytes), sizeof(int));
    file.write(reinterpret_cast<const char*>(compressed), written_bytes);

    cookie = 0x68;
    file.write(reinterpret_cast<const char*>(&cookie), sizeof(cookie));

    float percentage = (static_cast<float>(written_bytes) / static_cast<float>(length)) * 100.0f;

    std::cout << "Wrote compressed, " << percentage << "% of original. ( " << length << " -> " << written_bytes << " )" << std::endl;
    if(stats_file) {
        *stats_file << length << "," << written_bytes << "," << percentage << ",";
    }

    delete[] compressed;

    return written_bytes;
}

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

EncodingSequence::~EncodingSequence() {
    std::cout << "Encoding Sequence destruct: " << this->from << " " << this->to << std::endl;
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
    if(!this->args->getShouldSkipReduction()) {
        for(int i = 3; i < OCTREE_MAX_DEPTH; i++) {
            std::cout << "------------------------------Deduplicating " << i << std::endl;
            this->deduplicator = new DeDuplicator(this->hashmaps[i], layeredContainer, i, this->args);
            this->deduplicator->run();
            delete this->deduplicator; //TODO
        }
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
    int nodeSize = sizeof(uint8_t) * 3 + 2 * sizeof(uint8_t);

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

    unsigned char quantization_start = this->args->getQuantizationStart();
    unsigned char quantization_end = this->args->getQuantizationEnd();

    build_quantization_lookup_table(quantization_start, quantization_end);
    file.write(reinterpret_cast<char*>(&quantization_start), sizeof(quantization_start));
    file.write(reinterpret_cast<char*>(&quantization_end), sizeof(quantization_end));

    int currentLayerOffset = 0;
    for(int i = 0; i < OCTREE_MAX_DEPTH; i++) {
        //file.write(reinterpret_cast<char*>(&currentLayerOffset), sizeof(int));
        int layerSize = nodeCount[i];
        file.write(reinterpret_cast<char*>(&layerSize), sizeof(int));
        currentLayerOffset += nodeSize*nodeCount[i] + childPtrCount[i] * sizeof(layer_ptr_type);
    }

    // Prepare statistics file if any
    std::ofstream* stats_file = nullptr;
    if(this->args->getCompressionStatsOutput() != nullptr) {
        stats_file = new std::ofstream();
        char filename[1024];
        snprintf(filename, sizeof(filename), "%s.%d-%d", this->args->getCompressionStatsOutput(), this->from, this->to);
        stats_file->open(filename, std::ios::out);
    }

    // Write the payload
    // By writing the bottom layers first, we can write the file in one go, 
    // even with having to take into account that child pointers change due to node trimming
    for(int i = OCTREE_MAX_DEPTH-1; i >= 0 ; i--) {
        if(stats_file) {
            *stats_file << i << ",";
        }
        this->writeLayerToDisk(trees, file, i, nodeCount[i], childPtrCount[i], stats_file);
        if(stats_file) {
            *stats_file << std::endl;
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

void EncodingSequence::writeLayerToDisk(LayeredOctreeProcessingContainer<octreeColorType>& trees, std::ofstream& file, int i, int nodeCount, int childPtrCount, std::ofstream* stats_file) {
    unsigned char cookie = 0x13;
    file.write(reinterpret_cast<const char*>(&cookie), sizeof(char));

    std::cout << "Writing layer ( " << trees.getLayerSize(i) << " nodes, " << nodeCount << " after trim)" << i << std::endl;
    long layerDistCount = 0;
    int measurementCounts = 0;

    int measureChildCount = 0;
    int childMeasurements = 0;


    int writtenNodes = 0;

    // Reduce file size
    int lastChildIdx = 0;
    int* childPointers = new int[childPtrCount];
    int childPtrOffset = 0;

    // Data
    unsigned char* y_data = new unsigned char[nodeCount];
    unsigned char* u_data = new unsigned char[nodeCount];
    unsigned char* v_data = new unsigned char[nodeCount];

    unsigned char* child_flag_data = new unsigned char[nodeCount];
    unsigned char* leaf_flag_data = new unsigned char[nodeCount];

    int dataWrittenCount = 0;

    for(int x = 0; x < trees.getLayerSize(i); x++) {
        auto node = trees.getNode(i, x);
        auto payload = node->getPayload();
        if(!payload->trimmed) {
            payload->writtenOffset = writtenNodes++;
            //std::cout << "Node " << x << ": NOT TRIMMED, written at " << payload->writtenOffset << std::endl;
            // Write the payload, converting glm::vec3 to rgb bytes

            glm::vec3 rgb(
                static_cast<float>(payload->data.x*255.0f),
                static_cast<float>(payload->data.y*255.0f),
                static_cast<float>(payload->data.z*255.0f)
            );

            glm::vec3 yuv = rgb_to_yuv(rgb);
            
            /*
            file.write(reinterpret_cast<char*>(&r), sizeof(uint8_t));
            file.write(reinterpret_cast<char*>(&g), sizeof(uint8_t));
            file.write(reinterpret_cast<char*>(&b), sizeof(uint8_t));
            */
            y_data[dataWrittenCount] = static_cast<unsigned char>(yuv.x);
            u_data[dataWrittenCount] = static_cast<unsigned char>(yuv.y);
            v_data[dataWrittenCount] = static_cast<unsigned char>(yuv.z);

            //uint8_t childCount = node->getChildCount();
            uint8_t childFlags = node->getChildFlags();
            uint8_t leafFlags = node->getLeafFlags();

            child_flag_data[dataWrittenCount] = childFlags;
            leaf_flag_data[dataWrittenCount] = leafFlags;
            /*
            file.write(reinterpret_cast<char*>(&childCount), sizeof(uint8_t));
            file.write(reinterpret_cast<char*>(&leafFlags), sizeof(uint8_t));
            */
            dataWrittenCount++;

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
                    int calculated_offset = child_idx - lastChildIdx;

                    children++;
                    //file.write(reinterpret_cast<char*>(&child_idx), sizeof(layer_ptr_type));
                    childPointers[childPtrOffset++] = calculated_offset;
                    lastChildIdx = child_idx;
                }
            }
            // Write the children offsets
            measureChildCount += children;
            childMeasurements++;
        } else {
            //std::cout << "Node " << x << ": VERY TRIMMED" << std::endl;
        }
    }
    if(dataWrittenCount != 0) {
        // Dump some images
        /*
        if(i== 10) {
            for(int i = 0; i < dataWrittenCount/(8*8); i++) {
                dump_y_to_png(&y_data[i*8*8], 8, 8);
                dump_u_to_png(&u_data[i*8*8], 8, 8);
                dump_v_to_png(&v_data[i*8*8], 8, 8);
            }
        }*/
        // Write node data
        //file.write(reinterpret_cast<char*>(r_data), dataWrittenCount);
        std::cout << "Writing colors using DCT:" << std::endl;
        write_color_compressed(y_data, dataWrittenCount, file, stats_file);
        //file.write(reinterpret_cast<char*>(g_data), dataWrittenCount);
        write_color_compressed(u_data, dataWrittenCount, file, stats_file);
        //file.write(reinterpret_cast<char*>(b_data), dataWrittenCount);
        write_color_compressed(v_data, dataWrittenCount, file, stats_file);

        std::cout << "Done writing colors" << std::endl;

        //file.write(child_count_data, dataWrittenCount);
        //write_compressed(child_count_data, dataWrittenCount, file);
        write_compressed(child_flag_data, dataWrittenCount, file, stats_file);
        //write_compressed(leaf_flag_data, dataWrittenCount, file, stats_file);
        //file.write(leaf_flag_data, dataWrittenCount);
    } else {
        std::cout << "No elements in layer " << i << " - skipping." << std::endl;
    }
    if(stats_file) {
        *stats_file << "|,";
    }
    // Run time length encode
    if(childPtrCount != 0) {
        int written_bytes = write_compressed(reinterpret_cast<unsigned char*>(childPointers), childPtrCount*sizeof(int), file, stats_file);

        //int outSize = srlec32((const unsigned char*)childPointers, childPtrOffset*sizeof(int), (unsigned char*)childRle);
        float percentage = (static_cast<float>(written_bytes) / static_cast<float>(childPtrOffset*sizeof(int))) * 100.0f;
        std::cout << "compression reduced layer ptr size from " << (childPtrOffset*sizeof(int)) << " to " << written_bytes << " ( " << percentage << " %)." << std::endl;
    } else {
        if(stats_file) {
            *stats_file << "0,0,0,";
        }
        std::cout << "Skipping child ptr for layer " << i << " due no children" << std::endl;
    }
    delete[] childPointers; 

    delete[] y_data;
    delete[] u_data;
    delete[] v_data;

    delete[] child_flag_data;
    //delete[] leaf_flag_data;

    // Verify that we didn't write more nodes than we calculated
    if(writtenNodes > nodeCount) {
        throw std::runtime_error("Wrote more nodes than expected...");
    }
    if(measurementCounts != 0) {
        std::cout << "Average distance between nodes: " << (layerDistCount / measurementCounts) << std::endl;
    }
    if(childMeasurements != 0) {
        std::cout << "Average child count: " << (measureChildCount / childMeasurements) << "( total " << measureChildCount << " )" << std::endl;
    }
}