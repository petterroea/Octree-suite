#include <pthread.h>
#include <semaphore.h>

#include <glm/vec3.hpp>

#include <vector>

struct Pointcloud {
    glm::vec3* points;
    glm::vec3* colors;
    int count;
};

class AsyncPointcloudWriter {
    pthread_t hThread;

    sem_t jobStartSemaphore;
    sem_t jobFinishedSemaphore;

    std::vector<Pointcloud> job;

    int writeCount = 0;

    static void threadEntrypoint(AsyncPointcloudWriter* me);
    void writeThread();
public:
    AsyncPointcloudWriter();
    ~AsyncPointcloudWriter();

    void write(std::vector<Pointcloud> pointclouds);
    void waitForSafeToWrite();
};