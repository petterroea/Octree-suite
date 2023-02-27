#include "octreeGenerator.h"

#include <iostream>


Octree<std::vector<Point>>* OctreeGenerator::boxSortOuter(int maxLevel) {
    auto octree = new Octree<std::vector<Point>>(std::vector<Point>());

    for(int entry = 0; entry < this->currentPointcloud->getPointCount(); entry++) {
        const glm::vec3 vertex = this->currentPointcloud->getVertices()[entry];

        int childIdx = ADDRESS_OCTREE_BY_VEC3(vertex);
        // Initialize the child if never done before
        if(octree->getChildByIdx(childIdx) == nullptr) {
            std::vector<Point> childPayload;
            childPayload.reserve(this->currentPointcloud->getPointCount());
#ifdef OCTREE_LOG
            std::cout << std::string(level*2, ' ') << "new child " << childIdx << " level " << level << " with max " << childPayload.max << std::endl;
#endif
            auto newChild = new Octree<std::vector<Point>>(childPayload);

            octree->setChild(newChild, childIdx);
        }
        auto child = octree->getChildByIdx(childIdx);
        // Translate the point to a coordinate system relative to the child
        // TODO: Maybe instead, we pass a transformation matrix into the boxSort algorithm so sign is kept?
        // Would be faster...
        auto transformedPoint = vertex*2.0f
            -glm::vec3(
                copysign(1.0f, vertex.x), 
                copysign(1.0f, vertex.y), 
                copysign(1.0f, vertex.z)
            );
        child->getPayload()->push_back(Point {
            transformedPoint,
            entry
        });
    }
    // We have copied over and transformed vertices, so now we recursively box sort the children
    for(int i = 0; i < 8; i++) {
        if(octree->getChildByIdx(i) != nullptr) {
            this->boxSort(octree->getChildByIdx(i), 1, maxLevel);
        }
    }
    return octree;
}

int FILL_DEPTH=7;
void OctreeGenerator::boxSort(Octree<std::vector<Point>>* node, int level, int maxLevel) {
    if(level == maxLevel) {
//#ifdef OCTREE_LOG
        std::cout << std::string(level*2, ' ') << "Hit max tree level, there are " << node->getPayload()->size() << " entries" << std::endl;
//#endif
        return;
    }
    if(node->getPayload()->size() <2 && level > FILL_DEPTH) {
#ifdef OCTREE_LOG
        std::cout << std::string(level*2, ' ') << "Hit leaf node level " << level << std::endl;
#endif
        return;
    }
    auto nodePayload = node->getPayload();
    for(auto listEntry : *nodePayload) {
        // Determine what child it belongs to
        glm::vec3 vertex = listEntry.xyz;
        int childIdx = ADDRESS_OCTREE_BY_VEC3(vertex);
        // Initialize the child if never done before
        if(node->getChildByIdx(childIdx) == nullptr) {
            std::vector<Point> childPayload;
            childPayload.reserve(nodePayload->size());
#ifdef OCTREE_LOG
            std::cout << std::string(level*2, ' ') << "new child " << childIdx << " level " << level << " with max " << childPayload.max << std::endl;
#endif

            auto newChild = new Octree<std::vector<Point>>(childPayload);

            node->setChild(newChild, childIdx);
        }

        auto child = node->getChildByIdx(childIdx);
        // Translate the point to a coordinate system relative to the child
        // TODO: Maybe instead, we pass a transformation matrix into the boxSort algorithm so sign is kept?
        // Would be faster...
        auto transformedPoint = vertex*2.0f
            -glm::vec3(
                copysign(1.0f, vertex.x), 
                copysign(1.0f, vertex.y), 
                copysign(1.0f, vertex.z)
            );
        child->getPayload()->push_back(Point {
            transformedPoint,
            listEntry.colorIdx
        });
    }
    //Recursively sort children
    for(int i = 0; i < 8; i++) {
        auto child = node->getChildByIdx(i);
        if(child != nullptr) {
#ifdef OCTREE_LOG
            std::cout << std::string(level*2, ' ') << "level " << level << " enter child " << i << " count " << child->getPayload()->count << std::endl;
#endif
            this->boxSort(child, level+1, maxLevel);
        }
    }
}
// Returns average color
glm::vec3 OctreeGenerator::serialize(Octree<std::vector<Point>>* node, std::ofstream &treefile, int* writeHead, int* nodeLocation) {
    // Different count from what the octree reports, as we ignore empty children
    unsigned char childCount = 0;
    unsigned char childFlags = 0;
    int offsets[8];

    glm::vec3 avgColor = glm::vec3(0.0f, 0.0f, 0.0f);

    // Is this a leaf node?
    bool leaf = false;
    if(node->getChildCount() == 0) {
        if(node->getPayload()->size() != 1) {
            std::cout << "aaa count is " << node->getPayload()->size() << std::endl;
        }
        avgColor = this->currentPointcloud->getColors()[(*node->getPayload())[0].colorIdx];
        leaf = true;
    } else {
        for(int i = 0; i < 8; i++) {
            childFlags = childFlags << 1;
            auto child = node->getChildByIdx(i);
            // count=0 means an empty bucket. It has colored siblings, but is itself empty air
            if(child != nullptr && child->getPayload()->size() > 0) {
                avgColor += this->serialize(child, treefile, writeHead, &offsets[childCount++]);
                childFlags = childFlags | 1;
            } 
        }

        if(childCount == 0) {
            std::cout << "NO CHILDREN, reported " << +node->getChildCount() << std::endl;
        }

        // This will look muddy
        avgColor = avgColor / static_cast<float>(childCount);
    }

    unsigned char r = static_cast<unsigned char>(avgColor.x*255.0f);
    unsigned char g = static_cast<unsigned char>(avgColor.y*255.0f);
    unsigned char b = static_cast<unsigned char>(avgColor.z*255.0f);

    *nodeLocation = *writeHead;
    //treefile.write(leaf ? bbb : aaa,  1);
    treefile.write((char*)&childCount, sizeof(unsigned char));
    treefile.write((char*)&childFlags, sizeof(unsigned char));
    treefile.write((char*)&r, sizeof(unsigned char));
    treefile.write((char*)&g, sizeof(unsigned char));
    treefile.write((char*)&b, sizeof(unsigned char));
    treefile.write((char*)offsets, sizeof(int)*childCount);
    *writeHead += (sizeof(unsigned char)*5+sizeof(int)*childCount);
    return avgColor;
}


void OctreeGenerator::writeToFile(Octree<std::vector<Point>>* octree, char* filename) {
    std::ofstream treeFile;
    treeFile.open(filename, std::ios::binary);

    unsigned int magic = 0xdeadbeef;
    treeFile.write((char*)&magic, sizeof(unsigned int));

    char version = 1;
    treeFile.write(&version, sizeof(char));

    int writeHead = 0;
    int rootLocation = 0;
    // Write an int so the root location space is allocated in the file
    // We will later re-seek to this point and overwrite the value
    treeFile.write((char*)&rootLocation, sizeof(int));

    this->serialize(octree, treeFile, &writeHead, &rootLocation);

    // Write where the root is
    treeFile.seekp(5);
    treeFile.write((char*)&rootLocation, sizeof(int));

    treeFile.close();
}

OctreeGenerator::OctreeGenerator(Pointcloud* currentPointcloud): currentPointcloud(currentPointcloud) {

}