#include <string>
#include <glm/vec3.hpp>

#include "../lib/octree.h"

Octree<glm::vec3>* loadOctree(std::string filename);