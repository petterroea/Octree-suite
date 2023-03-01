#include <octree/pointerOctree.h>
#include <glm/vec3.hpp>

#include <vector>
#include <fstream>

#include <math.h>

#include "pointcloud.h"

#define ADDRESS_OCTREE_BY_VEC3(vec) ADDRESS_OCTREE(signbit(vec.x), signbit(vec.y), signbit(vec.z))

struct Point {
    glm::vec3 xyz;
    int colorIdx; // The color doesn't change, so just keep the color index
};

class OctreeGenerator {
    Pointcloud* currentPointcloud;

    void boxSort(PointerOctree<std::vector<Point>>* node, int level, int maxLevel);
    glm::vec3 serialize(PointerOctree<std::vector<Point>>* node, std::ofstream &treefile, int* writeHead, int* nodeLocation);
public:
    OctreeGenerator(Pointcloud* currentPointcloud);

    PointerOctree<std::vector<Point>>* boxSortOuter(int maxLevel);

    void writeToFile(PointerOctree<std::vector<Point>>* octree, char* filename);
};