#include "pointcloud.h"

#include <iostream>
#include <exception>
#include <filesystem>

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
    unsigned char* values = new unsigned char[mapping.size];
    std::cout << "Parsing " << count << " ply values... (ply size " << mapping.size << ", number: " << count << " )" << std::endl;
    for(int i = 0; i < count; i++) {
        handle.read((char*)values, mapping.size);
        points[i].x = *((float*)(&values[mapping.x]));
        points[i].y = *((float*)(&values[mapping.y]));
        points[i].z = *((float*)(&values[mapping.z]));

        colors[i].x = static_cast<float>(values[mapping.r])/255.0f;
        colors[i].y = static_cast<float>(values[mapping.g])/255.0f;
        colors[i].z = static_cast<float>(values[mapping.b])/255.0f;
    }
    delete[] values;
    std::cout << "Successfully parsed ply" << std::endl;
    return new Pointcloud(points, colors, count);
}

Pointcloud* parsePlyFile(std::string filename) {
    std::ifstream handle(filename);
    std::cout << "Loading " << filename << std::endl;
    if(!handle.is_open()) {
        std::cout << "Failed to open file" << std::endl;
        throw std::runtime_error("Failed to open file");
    }
    std::string line;
    getline(handle, line);
    if(line != "ply") {
        std::cout << "Invalid magic" << std::endl;
        throw std::runtime_error("Invalid magic");
    }

    getline(handle, line);
    PlyType type = determinePlyType(line);
    if(type == PlyType::UNKNOWN) {
        std::cout << "Unknown PLY file" << std::endl;
        throw std::runtime_error("Unknown PLY file");
    }

    std::string element("element ");
    // Parse the element header
    getline(handle, line);
    if(line.rfind(element) != 0) {
        std::cout << "Failed to parse element header" << std::endl;
        throw std::runtime_error("Expected element");
    }
    line = line.substr(line.rfind(" ") + 1);
    int elementCount = std::atoi(line.c_str());
    std::cout << elementCount << " elements" << std::endl;

    PlyMapping mapping;
    std::string floatName("float ");
    std::string ucharName("uchar ");
    std::string property("property ");
    int curIdx = 0;
    // Parse the header until we reach end_header
    while(true) {
        getline(handle, line);
        if(line.rfind(property) != 0) {
            if(line.rfind(element) == 0) {
                std::cout << "More than one element - unsupported" << std::endl;
                throw std::runtime_error("More than one element - unsupported file");
            }
            else if(line == "end_header") {
                break;
            } else {
                std::cout << "Unsupported line" << std::endl;
                throw std::runtime_error("Unsupported line");
            }
        } else {
            line = line.substr(property.length());
            if(line.rfind(floatName) != 0 && line.rfind(ucharName) != 0) {
                std::cout << "Invalid data type: " << line << std::endl;
                throw std::runtime_error("Invalid data type - not supported");
            }
            line = line.substr(floatName.length());
            // Determine property name
            if(line == "x") {
                mapping.x = curIdx;
                curIdx += 4;
            } else if(line == "y") {
                mapping.y = curIdx;
                curIdx += 4;
            } else if(line == "z") {
                mapping.z = curIdx;
                curIdx += 4;
            } else if(line == "red") {
                mapping.r = curIdx++;
            } else if(line == "green") {
                mapping.g = curIdx++;
            } else if(line == "blue") {
                mapping.b = curIdx++;
            } else {
                std::cout << "Unknown property type: " << line << std::endl;
                throw std::runtime_error("Unknown property type " + line);
            }
        }
    }
    mapping.size = curIdx;
    std::cout << "Parsing ply..." << std::endl;

    // Sanity check the file size
    if(std::filesystem::file_size(filename) < curIdx * elementCount) {
        std::cout << "ERROR: Unable to parse pointcloud - expected size is bigger than the file" << std::endl;
        throw std::runtime_error("Invalid pointcloud file");
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