#include "octreeFrameset.h"


OctreeFrameset::OctreeFrameset(int startIndex, int endIndex, std::string filename) : startIndex(startIndex), endIndex(endIndex), filename(filename) {

}

int OctreeFrameset::getStartIndex() const {
    return this->startIndex;
}

int OctreeFrameset::getEndIndex() const {
    return this->endIndex;
}
std::string OctreeFrameset::getFilename() const {
    return this->filename;
}