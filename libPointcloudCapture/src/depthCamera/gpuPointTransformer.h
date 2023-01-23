#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <cuda_runtime.h>
#include "types.h"

class GpuPointTransformer {
    // These buffers are used to do transformation to world-space on the GPU
    // We then copy the resulting points to RAM using double buffering
    // Double buffering allows us to write to disk while the next frame is being copied back
    void* devPtrPointsTransformed = nullptr;
    glm::vec3* hostPointsTransformed[2] { nullptr, nullptr };

    void* devPtrPointColors = nullptr;
    glm::vec3* hostColors[2] { nullptr, nullptr };

    int currentBuffer = 0;
public:
    GpuPointTransformer(VideoMode mode);
    ~GpuPointTransformer();
    void transformPoints(void* pointBuffer, cudaTextureObject_t texture, void* textureCoords, int pointCount, glm::mat4x4 transform);
    void getBuffers(glm::vec3** points, glm::vec3** colors);
    void swapBuffers();
};