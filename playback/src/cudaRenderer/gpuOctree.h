#include <glm/vec3.hpp>

// TODO rename GpuPointerOctree?
class GpuOctree {
public:
    int children[8] {
        0,0,0,0,0,0,0,0
    };
    glm::vec3 color;
    GpuOctree(glm::vec3 color);
};