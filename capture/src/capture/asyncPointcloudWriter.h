#include <pthread.h>
#include <semaphore.h>

#include <glm/vec3.hpp>

#include <vector>

struct Pointcloud {
    glm::vec3* points;
    glm::vec3* colors;
    int count;
};

struct Point {
    glm::vec3 pos;
    glm::vec3 color;
};

class AsyncPointcloudWriter {
    int maxExpectedPoints;
    Point* points;

    pthread_t hThread;

    sem_t jobStartSemaphore;
    sem_t jobFinishedSemaphore;

    std::vector<Pointcloud> job;

    int writeCount = 0;

    static void threadEntrypoint(AsyncPointcloudWriter* me);
    void writeThread();
public:
    AsyncPointcloudWriter(int maxExpectedPoints);
    ~AsyncPointcloudWriter();

    void write(std::vector<Pointcloud> pointclouds);
    void waitForSafeToWrite();
};