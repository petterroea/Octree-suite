#include "pointcloud.h"

#include <iostream>

Pointcloud::Pointcloud(glm::vec3* vertices, glm::vec3* colors, int pointCount): vertices(vertices), colors(colors), pointCount(pointCount) {

}

Pointcloud::~Pointcloud() {
    delete[] this->vertices;
    delete[] this->colors;
}

PlyType determinePlyType(std::string line) {
    std::string format("format ");
    if(line.rfind(format) != 0) {
        return PlyType::UNKNOWN;
    }
    line = line.substr(format.length());
    if(line == "ascii 1.0") {
        return PlyType::ASCII;
    }
    else if(line == "binary_little_endian 1.0") {
        return PlyType::BINARY_LITTLE_ENDIAN;
    }
    else if(line == "binary_big_endian 1.0") {
        return PlyType::BINARY_BIG_ENDIAN;
    }
    return PlyType::UNKNOWN;
}

Pointcloud* parseLittleEndianBinaryPly(PlyMapping& mapping, std::ifstream& handle, int count) {
    glm::vec3* points = new glm::vec3[count];
    glm::vec3* colors = new glm::vec3[count];
    for(int i = 0; i < count; i++) {
        float values[6];
        handle.read((char*)&values, sizeof(float)*6);
        points[i].x = values[mapping.x];
        points[i].y = values[mapping.y];
        points[i].z = values[mapping.z];

        colors[i].x = values[mapping.r];
        colors[i].y = values[mapping.g];
        colors[i].z = values[mapping.b];
    }
    return new Pointcloud(points, colors, count);
}

Pointcloud* parsePlyFile(std::string filename) {
    std::ifstream handle(filename);
    if(!handle.is_open()) {
        std::cout << "Failed to open file" << std::endl;
        throw "Failed to open file";
    }
    std::string line;
    getline(handle, line);
    if(line != "ply") {
        std::cout << "Invalid magic" << std::endl;
        throw "Invalid magic";
    }

    getline(handle, line);
    PlyType type = determinePlyType(line);
    if(type == PlyType::UNKNOWN) {
        std::cout << "Unknown PLY file" << std::endl;
        throw "Unknown PLY file";
    }

    std::string element("element ");
    // Parse the element header
    getline(handle, line);
    if(line.rfind(element) != 0) {
        std::cout << "Failed to parse element header" << std::endl;
        throw "Expected element";
    }
    line = line.substr(line.rfind(" ") + 1);
    int elementCount = std::atoi(line.c_str());
    std::cout << elementCount << " elements" << std::endl;

    PlyMapping mapping;
    std::string floatName("float ");
    std::string property("property ");
    int curIdx = 0;
    // Parse the header until we reach end_header
    while(true) {
        getline(handle, line);
        if(line.rfind(property) != 0) {
            if(line.rfind(element) == 0) {
                throw "More than one element - unsupported file";
            }
            else if(line == "end_header") {
                break;
            } else {
                throw "Unsupported line";
            }
        } else {
            line = line.substr(property.length());
            if(line.rfind(floatName) != 0) {
                throw "Invalid data type - not supported";
            }
            line = line.substr(floatName.length());
            // Determine property name
            if(line == "x") {
                mapping.x = curIdx++;
            } else if(line == "y") {
                mapping.y = curIdx++;
            } else if(line == "z") {
                mapping.z = curIdx++;
            } else if(line == "r") {
                mapping.r = curIdx++;
            } else if(line == "g") {
                mapping.g = curIdx++;
            } else if(line == "b") {
                mapping.b = curIdx++;
            } else {
                throw "Unknown property type";
            }
        }
    }

    if(!mapping.valid()) {
        throw "Invalid mapping";
    }
    if(type != PlyType::BINARY_LITTLE_ENDIAN) {
        throw "Unsupported PLY file";
    } else {
        return parseLittleEndianBinaryPly(mapping, handle, elementCount);
    }

}