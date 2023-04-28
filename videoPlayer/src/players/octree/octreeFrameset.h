#pragma once

#include <string>

/*
 * Data structure representing a set of octree frames
*/
class OctreeFrameset {
    int startIndex;
    int endIndex;
    std::string filename;

public:
    OctreeFrameset(int startIndex, int endIndex, std::string filename);
    
    int getStartIndex() const;
    int getEndIndex() const;
    std::string getFilename() const;
};

