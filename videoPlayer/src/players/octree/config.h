#pragma once
#include <glm/vec3.hpp>
#include <layeredOctree/layeredOctreeContainerCuda.h>

typedef glm::vec3 octreeColorType;
typedef LayeredOctreeContainerCuda<octreeColorType> loadedOctreeType;
