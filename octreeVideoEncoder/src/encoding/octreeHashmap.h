#pragma once

#include <glm/vec3.hpp>
#include <octree/octree.h>

#include <vector>

struct octree_node {
    
};

typedef std::vector<int> hashmap_element;

#define HASHMAP_SIZE 256

// Represents a hashmap of indexes to octrees in a ChunkedOctreeContainer
class OctreeHashmap {
    hashmap_element* elements[HASHMAP_SIZE];
public:
    OctreeHashmap();
    ~OctreeHashmap();

    void push(int key, int element_idx);
    hashmap_element* get_vector(int key) const;

    void clear();

};