#include "asyncPointcloudWriter.h"

#include <fstream>
#include <chrono>
#include <iostream>
#include <filesystem>

AsyncPointcloudWriter::AsyncPointcloudWriter(std::filesystem::path outputDirectory, int maxExpectedPoints) : outputDirectory(outputDirectory), maxExpectedPoints(maxExpectedPoints) {
    //Pre-allocate a write buffer to save time when writing
    this->points = new Point[maxExpectedPoints];
    sem_init(&this->jobStartSemaphore, 0, 0);
    sem_init(&this->jobFinishedSemaphore, 0, 1);
    this->hThread = pthread_create(&this->hThread, NULL, (void* (*)(void*))AsyncPointcloudWriter::threadEntrypoint, (void*)this);
}

AsyncPointcloudWriter::~AsyncPointcloudWriter() {
    pthread_join(this->hThread, NULL);
    delete this->points;
}

void AsyncPointcloudWriter::threadEntrypoint(AsyncPointcloudWriter* me) {
    me->writeThread();
}
#define VALID_POINT(point) (point.x != 0.0f && \
                            point.y != 0.0f && \
                            point.z != 0.0f)

void AsyncPointcloudWriter::writeThread() {
    while(true) {
        std::cout << "Starting write" << std::endl;
        sem_wait(&this->jobStartSemaphore);
        auto start = std::chrono::system_clock::now();
        int vertexCount = 0;
        // TODO this could be faster
        for(auto pointcloud : this->job) {
            for(int i = 0; i < pointcloud.count; i++) {
                auto point = pointcloud.points[i];
                auto color = pointcloud.colors[i];
                if(VALID_POINT(point)) {
                    this->points[vertexCount].pos = point;
                    this->points[vertexCount].color = color;
                    vertexCount++;
                }
            }
        }
        char headerBuffer[1000];
        int headerLen = sprintf(headerBuffer, "ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty float r\nproperty float g\nproperty float b\nend_header\n", vertexCount);

        std::ofstream handle;

        std::filesystem::path filename("capture_" + std::to_string(this->writeCount) + ".ply");
        std::filesystem::path fullFilename = this->outputDirectory / filename;

        handle.open(fullFilename.string(), std::ios::binary);
        // Write the header
        handle.write(headerBuffer, headerLen);
        // Write the body
        handle.write((char*)this->points, vertexCount*sizeof(Point));

        this->writeCount++;
        handle.close();
        auto end = std::chrono::system_clock::now();
        float elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Saved pointcloud in " << elapsed_time << "ms." << std::endl;
        sem_post(&this->jobFinishedSemaphore);
    }
}

void AsyncPointcloudWriter::write(std::vector<Pointcloud> pointclouds) {
    std::cout << "Requesting write" << std::endl;
    job = pointclouds;
    sem_post(&this->jobStartSemaphore);
}
void AsyncPointcloudWriter::waitForSafeToWrite() {
    std::cout << "Waiting for it to be safe to write" << std::endl;
    sem_wait(&this->jobFinishedSemaphore);
}