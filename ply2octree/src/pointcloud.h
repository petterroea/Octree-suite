#pragma once
#include <glm/vec3.hpp>

#include <string>
#include <fstream>

enum PlyType {
    ASCII,
    BINARY_LITTLE_ENDIAN,
    BINARY_BIG_ENDIAN,
    UNKNOWN
};

// Allows us to handle PLY files with variable mappings
struct PlyMapping {
    int x = -1;
    int y = -1;
    int z = -1;
    int r = -1;
    int g = -1;
    int b = -1;
    
    bool valid() {
        return x != -1 && y != -1 && z != -1 && r != -1 && g != -1 && b != -1;
    }
};

class Pointcloud {
    glm::vec3* vertices;
    glm::vec3* colors;

    int pointCount;

protected:
    Pointcloud(glm::vec3* vertices, glm::vec3* colors, int pointCount);

    friend Pointcloud* parseLittleEndianBinaryPly(PlyMapping& mapping, std::ifstream& handle, int count);
public:
    ~Pointcloud();

    const glm::vec3* getVertices() { return this->vertices; }
    const glm::vec3* getColors() { return this->colors; }
    const int getPointCount() { return this->pointCount; }
};

Pointcloud* parsePlyFile(std::string filename);