#include <vector>
#include <mutex>

template <typename T>
class ParallelWorker {
    void (*callback)(T input);

    std::vector<T> tasks;
    std::mutex taskFetchMutex;
public:
    ParallelWorker(void (*callback)(T input), std::vector<T> tasks);

    void waitForFinish();
};

template <typename T>
ParallelWorker<T>::ParallelWorker(void (*callback)(T input), std::vector<T> tasks) {
    
}