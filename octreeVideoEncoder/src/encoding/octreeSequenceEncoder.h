#pragma once

#include <filesystem>
#include <queue>
#include <vector>
#include <thread>
#include <mutex>

#include "encodingJob.h"
#include "../octreeSequence.h"
#include "../videoEncoderRunArgs.h"

class OctreeSequenceEncoder {
    OctreeSequence* sequence;
    VideoEncoderRunArgs* args;
    std::filesystem::path outputFolder;
    std::queue<EncodingJob*> jobs;

    // Thread stuff
    std::vector<std::thread*> threadPool;
    std::mutex jobMutex;

    EncodingJob* getJob();

    static void worker(OctreeSequenceEncoder* me);

public:
    OctreeSequenceEncoder(OctreeSequence* sequence, std::filesystem::path outputFolder, VideoEncoderRunArgs* args);
    ~OctreeSequenceEncoder();

    void encode();
};