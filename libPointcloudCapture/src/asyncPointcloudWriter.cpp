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
                    // Convert color to RGB char
                    this->points[vertexCount].red = static_cast<unsigned char>(color.r * 255.0f);
                    this->points[vertexCount].green = static_cast<unsigned char>(color.g * 255.0f);
                    this->points[vertexCount].blue = static_cast<unsigned char>(color.b * 255.0f);
                    vertexCount++;
                }
            }
        }
        char headerBuffer[1000];
        int headerLen = sprintf(headerBuffer, "ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n", vertexCount);

        std::ofstream handle;

        char fullFilenameBuf[100];
        snprintf(fullFilenameBuf, sizeof(fullFilenameBuf), "capture_%06d.ply", this->writeCount);

        std::filesystem::path filename(fullFilenameBuf);
        std::filesystem::path fullFilename = this->outputDirectory / filename;

        handle.open(fullFilename.string(), std::ios::binary);
        // Write the header
        handle.write(headerBuffer, headerLen);
        // Write the body
        for(int i = 0; i < vertexCount; i++) {
            handle.write((char*)&this->points[i].pos, sizeof(glm::vec3));
            handle.write((char*)&this->points[i].red, sizeof(unsigned char));
            handle.write((char*)&this->points[i].green, sizeof(unsigned char));
            handle.write((char*)&this->points[i].blue, sizeof(unsigned char));
        }

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