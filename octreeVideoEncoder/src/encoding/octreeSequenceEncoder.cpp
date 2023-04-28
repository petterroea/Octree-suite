#include "octreeSequenceEncoder.h"
#include "encodingSequence.h"

#include <rapidjson/document.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/prettywriter.h>

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

OctreeSequenceEncoder::OctreeSequenceEncoder(OctreeSequence* sequence, std::filesystem::path outputFolder, VideoEncoderRunArgs* args) : sequence(sequence), outputFolder(outputFolder), args(args) {
    // Make sure we have an output dir
    if(!std::filesystem::exists(this->outputFolder)) {
        std::filesystem::create_directories(this->outputFolder);
    } else if (!std::filesystem::is_directory(this->outputFolder)) {
        throw std::runtime_error("Output folder is not a directory");
    } 
}

OctreeSequenceEncoder::~OctreeSequenceEncoder() {

}

// Thread worker
void OctreeSequenceEncoder::worker(OctreeSequenceEncoder* me) {
    EncodingJob* job = me->getJob();
    while(job != nullptr) {
        //std::cout << "Starting new job" << std::endl;

        // Populate metadata json
        auto sequence = new EncodingSequence(
            me->sequence, 
            job->getFrom(), 
            job->getTo(),
            job->getFullSequencePath().string(),
            me->args
        );
        sequence->encode();

        // Cleanup stuff we don't need
        delete sequence;
        delete job;

        job = me->getJob();
    }
}

EncodingJob* OctreeSequenceEncoder::getJob() {
    std::lock_guard<std::mutex>(this->jobMutex);
    if(this->jobs.empty()) {
        return nullptr;
    }
    EncodingJob* job = this->jobs.front();
    this->jobs.pop();
    return job;
}

void OctreeSequenceEncoder::encode() {
    std::filesystem::path metadataFile = outputFolder / "metadata.json";
    if(std::filesystem::exists(metadataFile)) {
        throw std::runtime_error("Metadata file already exists");
    }

    int limit = this->args->getFrameLimit() < 0 ? sequence->getFrameCount() : std::min(this->args->getFrameLimit(), sequence->getFrameCount());

    // Create a json document we fill with contents as we create the queue
    rapidjson::Document d;
    d.SetObject();
    rapidjson::Document::AllocatorType& allocator = d.GetAllocator();
    rapidjson::Value frameSequence(rapidjson::kArrayType);

    int currentAt = 0;
    while(currentAt <= limit - 1) {
        int length = std::min(
            limit - currentAt - 1, 
            this->args->getEncodingChunkSize());
        std::cout << "Next length: " << length << std::endl;

        auto from = currentAt;
        auto to = currentAt + length;

        std::string filename = std::to_string(from) + "-" + std::to_string(to) + std::string(".loc");

        std::filesystem::path fullSequencePath = outputFolder / filename;

        
        // Build encoding job
        EncodingJob* job = new EncodingJob(from, to, fullSequencePath);
        this->jobs.push(job);

        currentAt += length + 1;

        // Create the metadata object for this frame sequence
        rapidjson::Value frame(rapidjson::kObjectType);


        rapidjson::Value frameFilename;
        frameFilename.SetString(filename.c_str(), allocator);
        frame.AddMember("filename", frameFilename, allocator);

        rapidjson::Value start;
        start.SetInt(from);
        frame.AddMember("start", start, allocator);

        rapidjson::Value end;
        end.SetInt(to);
        frame.AddMember("end", end, allocator);

        frameSequence.PushBack(frame, allocator);
    }

    // Start encoding encoding threads
    for(int i = 0; i < this->args->getChunkConcurrencyCount(); i++) {
        this->threadPool.push_back(new std::thread(OctreeSequenceEncoder::worker, this));
    }

    for(auto thread : this->threadPool) {
        thread->join();
        delete thread;
    }

    d.AddMember("frameSequence", frameSequence, allocator);

    // Add some metadata
    {
        rapidjson::Value fps;
        fps.SetFloat(30.0f);

        rapidjson::Value key;
 
        key.SetString("fps");
        d.AddMember(key, fps, allocator);
    }
    {
        rapidjson::Value frameCount;
        frameCount.SetInt(sequence->getFrameCount());

        rapidjson::Value key;
 
        key.SetString("frameCount");
        d.AddMember(key, frameCount, allocator);
    }

    std::ofstream out(metadataFile.string());
    rapidjson::OStreamWrapper osw(out);
    rapidjson::PrettyWriter<rapidjson::OStreamWrapper> writer(osw);
    d.Accept(writer);

    std::cout << "Done writing metadata file, thank you come again!" << std::endl;
}