#include "asyncPointcloudWriter.h"

#include <fstream>
#include <chrono>
#include <iostream>

AsyncPointcloudWriter::AsyncPointcloudWriter() {
    sem_init(&this->jobStartSemaphore, 0, 0);
    sem_init(&this->jobFinishedSemaphore, 0, 1);
    this->hThread = pthread_create(&this->hThread, NULL, (void* (*)(void*))AsyncPointcloudWriter::threadEntrypoint, (void*)this);
}

AsyncPointcloudWriter::~AsyncPointcloudWriter() {
    pthread_join(this->hThread, NULL);
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
                if(VALID_POINT(point)) {
                    vertexCount++;
                }
            }
        }
        // Write the header
        std::ofstream handle;
        handle.open("capture_" + std::to_string(this->writeCount) + ".ply");
        handle << "ply" << std::endl;
        handle << "format ascii 1.0" << std::endl;
        handle << "element vertex " << vertexCount << std::endl;
        handle << "property float x" << std::endl;
        handle << "property float y" << std::endl;
        handle << "property float z" << std::endl;
        // TODO save filespace by saving as uchar?
        handle << "property float r" << std::endl;
        handle << "property float g" << std::endl;
        handle << "property float b" << std::endl;
        handle << "end_header" << std::endl;

        // Write the vertices
        for(auto pointcloud : this->job) {
            for(int i = 0; i < pointcloud.count; i++) {
                auto point = pointcloud.points[i];
                auto color = pointcloud.colors[i];
                if(VALID_POINT(point)) {
                    handle << point.x << " " << point.y << " " << point.z << " "
                    << color.x << " " << color.y << " " << color.z << std::endl;
                }
            }
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