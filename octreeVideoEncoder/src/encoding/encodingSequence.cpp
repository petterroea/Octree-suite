#include "encodingSequence.h"

#include <math.h>
#include <iostream>
#include <exception>
#include <thread>

#include <glm/vec3.hpp>

EncodingSequence::EncodingSequence(OctreeSequence* sequence, int from, int to): 
    sequence(sequence), 
    from(from), 
    to(to) {
    if(from < 0 || to > sequence->getFrameCount()) {
        throw std::invalid_argument("Invalid frame range");
    }
}

void EncodingSequence::encode() {
    Octree<glm::vec3>** octrees = new Octree<glm::vec3>*[this->to - this->from + 1];
    for(int frame = this->from; frame <= this->to; frame++) {
        std::cout << "Loading frame " << frame << std::endl;
        octrees[frame - this->from] = this->sequence->getOctree(frame);
        std::cout << "Octree " << frame << " fill rate: " << octreeFillRate(octrees[frame - this->from]) << std::endl;
    }
    std::cout << "Calculating deltas" << std::endl;
    for(int frame = this->from; frame <= this->to - 1; frame++) {
        auto lhs = octrees[frame - this->from];
        auto rhs = octrees[frame - this->from + 1];
        std::cout << "Color diff " << frame << " to " << (frame+1) << ": " << diffOctreeColor(lhs, rhs) << std::endl;
        std::cout << "Octree diff " << frame << " to " << (frame+1) << ": " << octreeSimilarity(lhs, rhs) << std::endl;
    }
    // Make the trees exist in the same context, then put them in a hashmap
    ChunkedOctreeContainer<glm::vec3> chunkedContainer(octrees[this->from]);

    for(int frame = this->from+1; frame <= this->to; frame++) {
        auto tree = octrees[frame - this->from];
        int rootIdx = chunkedContainer.addOctree(tree);
        std::cout << "Installed octree with rootidx " << rootIdx << std::endl;

        this->populateHashmap(0, rootIdx, chunkedContainer, 6);
    }

    // TODO better
    this->deduplicator = new DeDuplicator(this->hashmap, std::thread::hardware_concurrency());
    this->deduplicator->run();
    /*
    std::cout << "Hashmap occupancy:" << std::endl;
    for(int i = 0; i < 256; i++) {
        auto vector = this->hashmap.get_vector(i);

        std::cout << i << ": ";
        if(vector) {
            std::cout << vector->size() << std::endl;
            int k = std::max(2, static_cast<int>(vector->size() / 50));
            this->kMeans(i, vector->size()/100, 4);
        } else {
            std::cout << "none" << std::endl;
        }
    }
    // Extract structures that are sufficiently similar
    */

    // Re-assemble trees
    std::cout << "Cleaning up" << std::endl;
    for(int frame = this->from; frame <= this->to; frame++) {
        delete octrees[frame - this->from];
    }
    delete[] octrees;
}

void EncodingSequence::populateHashmap(int depth, int idx, ChunkedOctreeContainer<glm::vec3>& octreeContainer, int max_depth) {
    auto tree = octreeContainer.getNode(depth, idx);
    this->hashmap.push(tree->getHashKey(), tree);
    if(!max_depth) {
        return;
    }
    for(int i = 0; i < 8; i++) {
        auto child = tree->getChildByIdx(i);
        if(child) {
            this->populateHashmap(child, max_depth-1);
        }
    }
}
