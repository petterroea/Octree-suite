#include "octreeHashmap.h"

#include <cstring>
#include <cstdlib>
#include <iostream>

OctreeHashmap::OctreeHashmap() {
    memset(elements, 0, sizeof(elements));
    std::cout << "octree hashmap construct" << std::endl;
}

OctreeHashmap::~OctreeHashmap() {
    for(int i = 0; i < HASHMAP_SIZE; i++) {
        if(this->elements[i]) {
            delete this->elements[i];
        }
    }
}

void OctreeHashmap::push(int key, octree_type element) {
    if(!this->elements[key]) {
        this->elements[key] = new std::vector<octree_type>();
    }
    this->elements[key]->push_back(element);
}

hashmap_element* OctreeHashmap::get_vector(int key) const {
    return this->elements[key];
}

void OctreeHashmap::clear() {
    for(int i = 0; i < HASHMAP_SIZE; i++) {
        if(this->elements[i]) {
            delete this->elements[i];
        }
    }
    memset(elements, 0, sizeof(elements));
}
