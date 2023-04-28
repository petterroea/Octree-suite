#include "octreeLoad.h"

#include <iostream>
#include <fstream>

PointerOctree<glm::vec3>* getChild(char* data, unsigned int offset) {
    int offsets[8];

    char childCount = data[offset];
    char childFlags = data[offset+1];
    unsigned char r = data[offset+2];
    unsigned char g = data[offset+3];
    unsigned char b = data[offset+4];
    glm::vec3 color = glm::vec3(
        static_cast<float>(r)/255.0f,
        static_cast<float>(g)/255.0f,
        static_cast<float>(b)/255.0f
    );
    int* offsetArrayPtr = (int*)(&data[offset+5]);

    auto octree = new PointerOctree<glm::vec3>(color);

    int offsetCount = 0;
    for(int i = 7; i >= 0; i--) {
        // Is there a child here?
        if(childFlags & 1 == 1) {
            octree->setChild(getChild(data, offsetArrayPtr[childCount-1-offsetCount++]), i);
        }
        childFlags = childFlags >> 1;
    }
    return octree;
}

PointerOctree<glm::vec3>* loadOctree(std::string filename) {
    std::ifstream reader(filename, std::ios::binary);

    unsigned int magic;
    reader.read((char*)&magic, sizeof(unsigned int));

    if(magic != 0xdeadbeef) {
        std::cout << "Invalid magic" << std::endl;
        _Exit(1);
    }

    char version;
    reader.read(&version, 1);
    if(version != 1) {
        std::cout << "Invalid octree version" << std::endl;
        _Exit(1);
    }

    unsigned int rootOffset;
    reader.read((char*)&rootOffset, sizeof(unsigned int));

    reader.seekg(0, std::ios::end);
    unsigned int fileSize = reader.tellg();

    char* fileBuffer = new char[fileSize];
    reader.seekg(9);
    reader.read(fileBuffer, fileSize);

    PointerOctree<glm::vec3>* root = getChild(fileBuffer, rootOffset);
    reader.close();

    delete[] fileBuffer;
    return root;
}