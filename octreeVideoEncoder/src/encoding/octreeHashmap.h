#pragma once

#include <glm/vec3.hpp>
#include <octree/octree.h>

#include <vector>

struct octree_node {
    
}

typedef Octree<glm::vec3>* octree_type;
typedef std::vector<octree_type> hashmap_element;

#define HASHMAP_SIZE 256

class OctreeHashmap {
    hashmap_element* elements[HASHMAP_SIZE];
public:
    OctreeHashmap();
    ~OctreeHashmap();

    void push(int key, octree_type element);
    hashmap_element* get_vector(int key) const;

    void clear();

};