#include <pthread.h>
#include <semaphore.h>

#include <glm/vec3.hpp>

#include <vector>
#include <string>
#include <filesystem>

struct Pointcloud {
    glm::vec3* points;
    glm::vec3* colors;
    int count;
};

struct Point {
    glm::vec3 pos;
    unsigned char red;
    unsigned char green;
    unsigned char blue;
};

class AsyncPointcloudWriter {
    int maxExpectedPoints;
    Point* points;

    pthread_t hThread;

    sem_t jobStartSemaphore;
    sem_t jobFinishedSemaphore;

    std::vector<Pointcloud> job;
    std::filesystem::path outputDirectory;

    int writeCount = 0;

    static void threadEntrypoint(AsyncPointcloudWriter* me);
    void writeThread();
public:
    AsyncPointcloudWriter(std::filesystem::path outputDirectory, int maxExpectedPoints);
    ~AsyncPointcloudWriter();

    void write(std::vector<Pointcloud> pointclouds);
    void waitForSafeToWrite();
};