#include "deduplicator.h"

#include <iostream>
#include <vector>
#include <algorithm>

#include <octree/octree.h>

#include <layeredOctree/treeComparison.h>

#include "../kernels/deduplicatorCuda.h"

DeDuplicator::DeDuplicator(OctreeHashmap& hashmap, LayeredOctreeProcessingContainer<octreeColorType>& container, int layer, VideoEncoderRunArgs* args) : hashmap(hashmap), container(container), args(args), layer(layer) {
    for(int i = 0; i < 256; i++) {
        auto job = new DeDuplicationJob();
        //printf("Created job %p\n", job);

        auto vector = this->hashmap.get_vector(i);

        if(vector) {
            job->count = vector->size();
        } else {
            job->count = 0;
        }
        job->jobId = i;

        this->jobs.push_back(job);
    }
    std::sort(jobs.begin(), jobs.end(), [](const DeDuplicationJob* a, const DeDuplicationJob* b) -> bool {
        return a->count > b->count;
    });

    this->jobsIterator = this->jobs.begin();
    //this->cudaContainer = new LayeredOctreeContainerCuda(this->container);
}

DeDuplicator::~DeDuplicator() {
    std::cout << "DeDuplicator destructor layer " << layer << std::endl;
    for(auto job : this->jobs) {
        delete job;
    }
}

void DeDuplicator::worker(DeDuplicator* me) {
    auto job = me->getNextJob();
    //std::cout << "Starting thread worker" << std::endl;
    while(job != nullptr) {
        int jobId = job->jobId;
        //std::cout << "Starting job " << jobId << std::endl;
        auto vector = me->hashmap.get_vector(jobId);

        //std::cout << "Doing " << jobId << " (" << job->count << ") " << ": " << std::endl;
        if(vector) {
            //std::cout << vector->size() << std::endl;
            int k = std::max(2, static_cast<int>(vector->size() / 50));
            me->kMeans(jobId, k, 4);
        } else {
            //std::cout << "none" << std::endl;
        }
        job = me->getNextJob();
    }
    //std::cout << "Out of work, quitting" << std::endl;
}

void DeDuplicator::run() {
    //std::cout << "Starting " << nThreads << " work threads" << std::endl;
    for(int i = 0; i < this->args->getEncodingThreadCount(); i++) {
        this->threadPool.push_back(new std::thread(DeDuplicator::worker, this));
    }
    for(auto thread : this->threadPool) {
        thread->join();
        delete thread;
    }
    //std::cout << "Finished number crunching" << std::endl;
}

DeDuplicationJob* DeDuplicator::getNextJob() {
    std::lock_guard<std::mutex>(this->jobMutex);

    if(this->jobsIterator == this->jobs.end()) {
        return nullptr;
    }

    auto job = *(this->jobsIterator++);
    //std::cout << "Got new deduplication job: " << job->jobId << std::endl;
    if(job->jobId > 256) {
        throw std::runtime_error("data corruption detected");
    }
    return job;
}

void DeDuplicator::kMeans(int key, int k, int steps) {
    // All nodes in the hashmap bucket
    std::vector<layer_ptr_type>& population = *this->hashmap.get_vector(key);
    //std::cout << "Kmeans k=" << k << " pop " << population.size() << std::endl;

    int populationSize = population.size();

    // Store the list of what nodes are trimmed here to speed things up
    bool* trimmedTable = new bool[populationSize];
    int nonTrimmedPopSize = 0;

    for(int x = 0; x < populationSize; x++) {
        auto payload = this->container.getNode(this->layer, population[x])->getPayload();
        trimmedTable[x] = payload->trimmed;
        if(!payload->trimmed) {
            nonTrimmedPopSize++;
        }
    }

    // Do we have enough non-trimmed population?
    if(nonTrimmedPopSize < 2) {
        delete[] trimmedTable;
        return;
    }
    // Otherwise, is k larger than the population?
    else if(nonTrimmedPopSize < k) {
        k = nonTrimmedPopSize;
    }

    int* centers = new int[k];
    // Pick random initial centers
    // This is an elaborate algorithm to ensure we pick a center that is not used for another value of k,
    // and that isn't trimmed.
    for(int i = 0; i < k; i++) {
        //std::cout << "Looking for k=" << i << std::endl;
        // Make sure to never pick something that is trimmed
        int randomIndex = rand() % population.size();
        int foundCenter = -1;
        for(int x = 0; x < population.size(); x++) {
            int center = (randomIndex + x) % population.size();
            //std::cout << "Checking " << center << std::endl;
            // Is the center trimmed?
            if(!trimmedTable[center]) {
                // Is the point already selected somewhere?
                bool alreadyUsed = false;
                for(int y = 0; y < i; y++) {
                    if(centers[y] == center) {
                        //std::cout << "Already used for " << y << std::endl;
                        alreadyUsed = true;
                        break;
                    }
                }
                if(!alreadyUsed) {
                    foundCenter = center;
                    break;
                }
            }
        }
        if(foundCenter == -1) {
            std::cout << "Unable to find center for k=" + 
                std::to_string(k) + 
                " popcount " + 
                std::to_string(population.size()) + 
                " untrimmed count " +
                std::to_string(nonTrimmedPopSize) << std::endl;
            throw std::runtime_error("fuck");
            //No candidate for this
        }
        centers[i] = foundCenter;
    }
    //std::cout << "Done" << std::endl;

    // Cluster assignments
    int* assignments = new int[population.size()]; // The cluster assignment of each element
    std::vector<int>* k_children = new std::vector<int>[k]; // The elements in each assignment
    for(int i = 0; i < k; i++) {
        k_children[i] = std::vector<int>();
    }

    // Aggregated sum of "bestness" of each element
    float* element_popularity = new float[population.size()];

    // Used to find the best node in n time instead of n*m
    float* best = new float[k];
    int* bestIndex = new int[k];

    float* nearnessTable = new float[populationSize * populationSize];

    //std::cout << "Building precalc nearness table" << std::endl;
    // TODO speed up with CUDA?
    if(false && populationSize > 200) {
        //std::cout << "Building precalc table on GPU (" << populationSize << ")" << std::endl;
        
        //buildSimilarityLookupTableCuda(nearnessTable, population, this->layer, cudaContainer);
    } else {
        //std::cout << "Building precalc table on CPU (" << populationSize << ")" << std::endl;
        for(int x = 0; x < populationSize; x++) {
            // If the node is trimmed we don't need to determine nearness
            if(!trimmedTable[x]) {
                if(x % 50 == 0) {
                    //std::cout << "X: " << x << std::endl;
                }
                for(int y = 0; y < populationSize; y++) {
                    //auto a = this->container.getNode(this->layer, population[x]);
                    //auto b = this->container.getNode(this->layer, population[y]);
                    float score = 0.0f; // 0 will never be considered
                    if(!trimmedTable[y]) { 
                        float nearness = layeredOctreeSimilarity<LayeredOctreeProcessingContainer<octreeColorType>>(population[x], population[y], this->layer, &this->container);
                        float colorSimilarity = diffProcessingLayeredOctreeColor<octreeColorType>(population[x], population[y], this->layer, &this->container);
                        //std::cout << "Color similarity: " << colorSimilarity << std::endl;
                        score = nearness / (1.0f + ( colorSimilarity * this->args->getColorImportanceFactor()));
                        //score = nearness;
                    } else {
                        //std::cout << "Y: Hit a hode that is already trimmed" << std::endl;
                    }
                    nearnessTable[x + y*populationSize] = score;
                }
            } else {
                //std::cout << "X: Hit a hode that is already trimmed" << std::endl;
                // Mark all rows as trimmed
                for(int y = 0; y < populationSize; y++) {
                    nearnessTable[x + y*populationSize] = 0.0f;
                }
            }
        }
    }
    //std::cout << "Precalc done" << std::endl;

    for(int step = 0; step < steps; step++) {
        //std::cout << "Step " << step << std::endl;
        k_children->clear();
        // Assignment
        //std::cout << "Assignment" << std::endl;
        for(int element_index = 0; element_index < population.size(); element_index++) {
            int closest_idx = 0;
            float closest = 0.0;

            if(!trimmedTable[element_index]) {
                for(int i = 0; i < k; i++) {
                    /*
                    float current_similarity = octreeSimilarity(
                        population[centers[i]], 
                        population[element_index]
                    );
                    */
                    float current_similarity = nearnessTable[centers[i] + element_index * populationSize];
                    if(current_similarity > closest) {
                        closest_idx = i;
                        closest = current_similarity;
                    }

                }
            }
            assignments[element_index] = closest_idx;
            k_children[closest_idx].push_back(element_index);
        }
        // Update
        // Calculate how much a given element "represents" its cluster
        //std::cout << "Update" << std::endl;
        for(int element_index = 0; element_index < population.size(); element_index++) {
            float similarity = 0.0f;
            int outer_assignment = assignments[element_index];

            if(!trimmedTable[element_index]) {
                for(int element_index_inner : k_children[outer_assignment]) {
                    // Every element, including itself, to speed things up
                    /*
                    similarity += octreeSimilarity(
                        population[element_index],
                        population[element_index_inner]
                    );
                    */
                similarity += nearnessTable[element_index + element_index_inner * populationSize];
                }
            }
            element_popularity[element_index] = similarity;
        }
        //std::cout << "Find best" << std::endl;
        // For each cluster, find the best representation
        for(int i = 0; i < k; i++) {
            best[i] = 0.0f;
            bestIndex[i] = 0;
        }
        for(int element_index = 0; element_index < population.size(); element_index++) {
            int assignment = assignments[element_index];

            //std::cout << "assignment " << assignment << std::endl;
            if(!trimmedTable[element_index] && best[assignment] < element_popularity[element_index]) {
                /*
                std::cout << "Element " << element_index << " (" << assignment << ") "
                          << " has pop " << element_popularity[element_index]
                          << " (better than " << best[assignment] << ")" << std::endl;
                */
                best[assignment] = element_popularity[element_index];
                bestIndex[assignment] = element_index;
            }
        }
        for(int i = 0; i < k; i++) {
            //std::cout << i << " set to " << bestIndex[i] << std::endl;
            centers[i] = bestIndex[i];
        }
    }

    // Trim nodes that are similar to the most popular node in the cluster
    for(int i = 0; i < population.size(); i++) {
        int bestIndex = centers[assignments[i]]; // Best node for the same cluster
        if(trimmedTable[bestIndex]) {
            throw new std::runtime_error("Best index for one of the k-clusters is trimmed...");
        }
        float current_similarity = nearnessTable[bestIndex + i * populationSize]; // Similarity between best and current
        // Is this node similar enough that we want to prune it?
        if(
            bestIndex != i && // Don't trim oneself
            current_similarity > this->args->getTreeNearnessFactor() && // Must be good enough
            !trimmedTable[i] // Don't trim a node that is already trimmed
        ) {
            //std::cout << "Trimming away!" << std::endl;
            std::cout << "Trimming " << i << " ( replacing with " << bestIndex << ")" << std::endl;
            this->container.getNode(this->layer, population[i])->getPayload()->replacement = population[bestIndex];
            this->markTreeAsTrimmed(this->layer, population[i]);
        }
    }

    delete[] centers;
    delete[] assignments;
    delete[] element_popularity;
    delete[] best;
    delete[] bestIndex;
    delete[] k_children;
    delete[] nearnessTable;
    delete[] trimmedTable;
}


void DeDuplicator::markTreeAsTrimmed(int layer, int index) {
    // Don't mark as trimmed if it has already been marked
    auto node = this->container.getNode(layer, index);
    auto payload = node->getPayload();
    if(payload->trimmed) {
        return;
    }
    payload->trimmed = true;
    for(int i = 0; i < OCTREE_SIZE; i++) {
        auto child = node->getChildByIdx(i);
        if(child != NO_NODE) {
            this->markTreeAsTrimmed(layer+1, node->getChildByIdx(i));
        }
    }
}