#include "deduplicator.h"

#include <iostream>
#include <vector>
#include <algorithm>

#include <octree/octree.h>

DeDuplicator::DeDuplicator(OctreeHashmap& hashmap, int nThreads) : hashmap(hashmap), nThreads(nThreads) {
    for(int i = 0; i < 256; i++) {
        auto job = new DeDuplicationJob;

        auto vector = this->hashmap.get_vector(i);

        if(vector) {
            job->count = vector->size();
        } else {
            job->count = 0;
        }
        job->jobId = i;

        jobs.push_back(job);
    }
    std::sort(jobs.begin(), jobs.end(), [](const DeDuplicationJob* a, const DeDuplicationJob* b) -> bool {
        return a->count > b->count;
    });
    this->currentJobIterator = jobs.begin();
}

DeDuplicator::~DeDuplicator() {
    for(auto thread : this->threadPool) {
        if(thread) {
            if(thread->joinable()) {
                thread->join();
            }
        }
        delete thread;
    }
    for(auto job : this->jobs) {
        delete job;
    }
}

void DeDuplicator::worker(DeDuplicator* me) {
    auto job = me->getNextJob();
    std::cout << "Starting thread worker" << std::endl;
    while(job != nullptr) {
        int jobId = job->jobId;
        auto vector = me->hashmap.get_vector(jobId);

        std::cout << "Doing " << jobId << " (" << job->count << ") " << ": " << std::endl;
        if(vector) {
            //std::cout << vector->size() << std::endl;
            int k = std::max(2, static_cast<int>(vector->size() / 50));
            me->kMeans(jobId, k, 4);
        } else {
            //std::cout << "none" << std::endl;
        }
        job = me->getNextJob();
    }
    std::cout << "Out of work, quitting" << std::endl;
}

void DeDuplicator::run() {
    std::cout << "Starting " << nThreads << " work threads" << std::endl;
    for(int i = 0; i < nThreads; i++) {
        this->threadPool.push_back(new std::thread(DeDuplicator::worker, this));
    }
    for(auto thread : this->threadPool) {
        thread->join();
    }
    std::cout << "Finished number crunching" << std::endl;
}

DeDuplicationJob* DeDuplicator::getNextJob() {
    std::lock_guard<std::mutex>(this->jobMutex);

    if(this->currentJobIterator == this->jobs.end()) {
        return nullptr;
    }
    auto job = *(this->currentJobIterator++);
    return job;
}

void DeDuplicator::kMeans(int key, int k, int steps) {
    std::vector<Octree<glm::vec3>*>& population = *this->hashmap.get_vector(key);
    //std::cout << "Kmeans k=" << k << " pop " << population.size() << std::endl;

    int* centers = new int[k];
    // Pick random initial centers
    for(int i = 0; i < k; i++) {
        centers[i] = rand() % population.size();
    }

    // Cluster assignments
    int* assignments = new int[population.size()]; // The assignment of each element
    std::vector<int>* k_children = new std::vector<int>[k]; // The elements in each assignment

    // Aggregated sum of "bestness" of each element
    float* element_popularity = new float[population.size()];

    // Used to find the best node in n time instead of n*m
    float* best = new float[k];
    int* bestIndex = new int[k];

    int populationSize = population.size();

    float* nearnessTable = new float[populationSize * populationSize];

    std::cout << "Building precalc nearness table" << std::endl;
    // TODO speed up with CUDA?
    for(int x = 0; x < populationSize; x++) {
        if(x % 50 == 0) {
            std::cout << "X: " << x << std::endl;
        }
        for(int y = 0; y < populationSize; y++) {
            auto a = population[x];
            auto b = population[y];
            float nearness = octreeSimilarity(a, b);
            nearnessTable[x + y*populationSize] = nearness;
        }
    }
    std::cout << "Precalc done" << std::endl;

    for(int step = 0; step < steps; step++) {
        //std::cout << "Step " << step << std::endl;
        k_children->clear();
        // Assignment
        //std::cout << "Assignment" << std::endl;
        for(int element_index = 0; element_index < population.size(); element_index++) {
            int closest_idx = 0;
            float closest = 0.0;
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
            assignments[element_index] = closest_idx;
            k_children[closest_idx].push_back(element_index);
        }
        // Update
        // Calculate how much a given element "represents" its cluster
        //std::cout << "Update" << std::endl;
        for(int element_index = 0; element_index < population.size(); element_index++) {
            float similarity = 0.0f;
            int outer_assignment = assignments[element_index];

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
            
            /*
            // OLD BAD LOOP
            for(int element_index_inner = 0; element_index_inner < population.size(); element_index_inner++) {
                // Calculate similarities for elements that are in the same cluster, but roll a dice
                if(assignments[element_index_inner] == outer_assignment &&
                    element_index_inner != element_index) {
                    similarity += octreeSimilarity(
                        population[element_index],
                        population[element_index_inner]
                    );
                }
            }
            */
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
            if(best[assignment] < element_popularity[element_index]) {
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

    // Find the most similar nodes in each cluster
    for(int i = 0; i < k; i++) {

    }

    delete[] centers;
    delete[] assignments;
    delete[] element_popularity;
    delete[] best;
    delete[] bestIndex;
    delete[] k_children;
    delete[] nearnessTable;
}