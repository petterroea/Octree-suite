#include "octreeCapture.h"

#include <librealsense2/rs.hpp>

#include <glm/gtc/matrix_transform.hpp>

#include <iostream>

#include "imgui.h"

OctreeCapture::OctreeCapture(): capturePosition(0.0f, 0.0f, 0.0f), lineRendererShader(), lineRenderer(&this->lineRendererShader) {
    // Axis lines
    this->lineRenderer.drawLine(
        glm::vec3(0.0f, 0.0f, 0.0f), 
        glm::vec3(1.0f, 0.0f, 0.0f), 
        glm::vec3(1.0f, 0.0f, 0.0f)
    );
    this->lineRenderer.drawLine(
        glm::vec3(0.0f, 0.0f, 0.0f), 
        glm::vec3(0.0f, 1.0f, 0.0f), 
        glm::vec3(0.0f, 1.0f, 0.0f)
    );
    this->lineRenderer.drawLine(
        glm::vec3(0.0f, 0.0f, 0.0f), 
        glm::vec3(0.0f, 0.0f, 1.0f), 
        glm::vec3(0.0f, 0.0f, 1.0f)
    );
    // Bottom box
    this->lineRenderer.drawLine(
        glm::vec3(-1.0f, 1.0f, -1.0f), 
        glm::vec3(-1.0f, 1.0f, 1.0f), 
        glm::vec3(1.0f, 1.0f, 1.0f)
    );
    this->lineRenderer.drawLine(
        glm::vec3(-1.0f, 1.0f, 1.0f), 
        glm::vec3(1.0f, 1.0f, 1.0f), 
        glm::vec3(1.0f, 1.0f, 1.0f)
    );
    this->lineRenderer.drawLine(
        glm::vec3(1.0f, 1.0f, 1.0f), 
        glm::vec3(1.0f, 1.0f, -1.0f), 
        glm::vec3(1.0f, 1.0f, 1.0f)
    );
    this->lineRenderer.drawLine(
        glm::vec3(1.0f, 1.0f, -1.0f), 
        glm::vec3(-1.0f, 1.0f, -1.0f), 
        glm::vec3(1.0f, 1.0f, 1.0f)
    );
    // Top box
    this->lineRenderer.drawLine(
        glm::vec3(-1.0f, -1.0f, -1.0f), 
        glm::vec3(-1.0f, -1.0f, 1.0f), 
        glm::vec3(1.0f, 1.0f, 1.0f)
    );
    this->lineRenderer.drawLine(
        glm::vec3(-1.0f, -1.0f, 1.0f), 
        glm::vec3(1.0f, -1.0f, 1.0f), 
        glm::vec3(1.0f, 1.0f, 1.0f)
    );
    this->lineRenderer.drawLine(
        glm::vec3(1.0f, -1.0f, 1.0f), 
        glm::vec3(1.0f, -1.0f, -1.0f), 
        glm::vec3(1.0f, 1.0f, 1.0f)
    );
    this->lineRenderer.drawLine(
        glm::vec3(1.0f, -1.0f, -1.0f), 
        glm::vec3(-1.0f, -1.0f, -1.0f), 
        glm::vec3(1.0f, 1.0f, 1.0f)
    );
    // Connecting lines
    this->lineRenderer.drawLine(
        glm::vec3(-1.0f, -1.0f, -1.0f), 
        glm::vec3(-1.0f, 1.0f, -1.0f), 
        glm::vec3(1.0f, 1.0f, 1.0f)
    );
    this->lineRenderer.drawLine(
        glm::vec3(-1.0f, -1.0f, 1.0f), 
        glm::vec3(-1.0f, 1.0f, 1.0f), 
        glm::vec3(1.0f, 1.0f, 1.0f)
    );
    this->lineRenderer.drawLine(
        glm::vec3(1.0f, -1.0f, -1.0f), 
        glm::vec3(1.0f, 1.0f, -1.0f), 
        glm::vec3(1.0f, 1.0f, 1.0f)
    );
    this->lineRenderer.drawLine(
        glm::vec3(1.0f, -1.0f, 1.0f), 
        glm::vec3(1.0f, 1.0f, 1.0f), 
        glm::vec3(1.0f, 1.0f, 1.0f)
    );
}

void OctreeCapture::displayGui(std::vector<PointcloudCameraRendererPair*> cameras) {
    ImGui::Begin("Capture settings");
    if(ImGui::CollapsingHeader("Bounds")) {
        ImGui::SliderFloat3("Offset", (float*)&this->capturePosition, -1.5f, 1.5f);
        ImGui::SliderFloat("Scale", &this->captureScale, 0.1f, 2.5f);
    }
    ImGui::Separator();
    if(ImGui::Button("Capture a frame")) {
        this->captureScene(cameras);
    }

    ImGui::End();
}

// This can be sped up quite a lot
void OctreeCapture::captureScene(std::vector<PointcloudCameraRendererPair*> cameras) {
    std::cout << "Starting capture" << std::endl;
    auto start = std::chrono::system_clock::now();
    // Calculate how much space we need
    int maxPoints = 0;
    for(auto it : cameras) {
        DepthCamera* camera = it->camera;
        rs2::points pointcloud = camera->getLastPointcloud();
        maxPoints += pointcloud.size();
    }

    Point* points = (Point*)malloc(maxPoints*sizeof(Point));
    int pointIdx = 0;

    for(auto it : cameras) {
        DepthCamera* camera = it->camera;
        rs2::points pointcloud = camera->getLastPointcloud();

        const rs2::vertex* vertices = pointcloud.get_vertices();
        const rs2::texture_coordinate* texcoords = pointcloud.get_texture_coordinates();

        rs2::video_frame colorFrame = camera->getLastFrame().get_color_frame();

        for(int i = 0; i < pointcloud.size(); i++) {
            glm::vec4 vectorWorld = camera->getCalibration()*glm::vec4(vertices[i].x, vertices[i].y, vertices[i].z, 1.0f);
            glm::vec3 transformed = glm::vec3(
                vectorWorld.x-this->capturePosition.x, 
                vectorWorld.y-this->capturePosition.y, 
                vectorWorld.z-this->capturePosition.z
            )/this->captureScale;
            if(abs(transformed.x) > 1.0f || abs(transformed.y) > 1.0f || abs(transformed.z) > 1.0f) {
                continue;
            }

            // Color sampling
            int img_x = static_cast<int>(texcoords[i].u*(float)colorFrame.get_width());
            int img_y = static_cast<int>(texcoords[i].v*(float)colorFrame.get_height());

            int image_idx = (img_x+img_y*colorFrame.get_width())*3;
            unsigned char r = ((unsigned char*)colorFrame.get_data())[image_idx];
            unsigned char g = ((unsigned char*)colorFrame.get_data())[image_idx+1];
            unsigned char b = ((unsigned char*)colorFrame.get_data())[image_idx+2];
            glm::vec3 sampled_color = glm::vec3(
                static_cast<float>(r)/255.0f,
                static_cast<float>(g)/255.0f,
                static_cast<float>(b)/255.0f
            );

            points[pointIdx++] = Point{
                .xyz = transformed,
                .rgb = sampled_color
            };
        }
    }

    auto sortFinish = std::chrono::system_clock::now();

    SizedArray<Point> rootPayload = SizedArray<Point>{
        .list = points,
        .count = pointIdx,
        .max = pointIdx
    };

    auto pointerGenFinish = std::chrono::system_clock::now();
    std::cout << "Captured " << pointIdx << " points out of " << maxPoints << " expected" << std::endl;

    Octree<SizedArray<Point>> root(rootPayload);
    this->boxSort(&root, 0, 20);

    auto boxSortFinish = std::chrono::system_clock::now();

    //Serialize and store
    std::ofstream treefile;
    treefile.open("octree.oct", std::ios::binary);

    unsigned int magic = 0xdeadbeef;
    treefile.write((char*)&magic, sizeof(unsigned int));

    char version = 1;
    treefile.write(&version, sizeof(char));
    // This will be filled in later

    int writeHead = 0;
    int rootLocation = 0;
    treefile.write((char*)&rootLocation, sizeof(int));

    this->serialize(&root, treefile, &writeHead, &rootLocation);

    // Write where the root is
    treefile.seekp(5);
    treefile.write((char*)&rootLocation, sizeof(int));

    treefile.close();

    std::cout << "Completed serialization, root is at " << rootLocation << std::endl;

    auto serializeFinish = std::chrono::system_clock::now();

    float sortTime = std::chrono::duration_cast<std::chrono::milliseconds>(sortFinish - start).count();
    float pointerGenTime = std::chrono::duration_cast<std::chrono::milliseconds>(pointerGenFinish- sortFinish ).count();
    float boxSortTime = std::chrono::duration_cast<std::chrono::milliseconds>(boxSortFinish - pointerGenFinish).count();
    float serializeTime = std::chrono::duration_cast<std::chrono::milliseconds>(serializeFinish- boxSortFinish).count();
    float totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(serializeFinish- start).count();
    std::cout << "Finished in " << totalTime << "ms (sort " << sortTime << "ms, pointer gen " << pointerGenTime << "ms, box sort " << boxSortTime << "ms, serialize " << serializeTime << "ms) " << std::endl;

    delete[] points;
}

glm::vec3 OctreeCapture::serialize(Octree<SizedArray<Point>>* node, std::ofstream &treefile, int* writeHead, int* nodeLocation) {
    // Different count from what the octree reports, as we ignore empty children
    unsigned char childCount = 0;
    unsigned char childFlags = 0;
    int offsets[8];

    glm::vec3 avgColor = glm::vec3(0.0f, 0.0f, 0.0f);

    // Is this a leaf node?
    bool leaf = false;
    if(node->getChildCount() == 0) {
        if(node->getPayload()->count != 1) {
            std::cout << "aaa count is " << node->getPayload()->count << std::endl;
        }
        avgColor = node->getPayload()->list[0].rgb;
        leaf = true;
    } else {
        for(int i = 0; i < 8; i++) {
            childFlags = childFlags << 1;
            auto child = node->getChildByIdx(i);
            // count=0 means an empty bucket. It has colored siblings, but is itself empty air
            if(child != nullptr && child->getPayload()->count > 0) {
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

int FILL_DEPTH=8;
void OctreeCapture::boxSort(Octree<SizedArray<Point>>* node, int level, int maxLevel) {
    if(level == maxLevel) {
//#ifdef OCTREE_LOG
        std::cout << std::string(level*2, ' ') << "Hit max tree level, there are " << node->getPayload()->count << " entries" << std::endl;
//#endif
        return;
    }
    if(node->getPayload()->count<2 && level > FILL_DEPTH) {
#ifdef OCTREE_LOG
        std::cout << std::string(level*2, ' ') << "Hit leaf node level " << level << std::endl;
#endif
        return;
    }
    auto nodePayload = node->getPayload();
    for(int i = 0; i < nodePayload->count; i++) {
        // Determine what child it belongs to
        int childIdx = ADDRESS_OCTREE_BY_VEC3(nodePayload->list[i].xyz);
        // Initialize the child if never done before
        if(node->getChildByIdx(childIdx) == nullptr) {
            SizedArray<Point> childPayload = SizedArray<Point>{
                .list = (Point*)malloc(nodePayload->count*sizeof(Point)),
                .count = 0,
                .max = nodePayload->count
            };

#ifdef OCTREE_LOG
            std::cout << std::string(level*2, ' ') << "new child " << childIdx << " level " << level << " with max " << childPayload.max << std::endl;
#endif

            auto newChild = new Octree<SizedArray<Point>>(childPayload);

            node->setChild(newChild, childIdx);
        }
        auto child = node->getChildByIdx(childIdx);
        auto payloadPtr = &child->getPayload()->list[child->getPayload()->count++];
        payloadPtr->rgb = nodePayload->list[i].rgb;
        // Translate the point to a coordinate system relative to the child
        // TODO: Maybe instead, we pass a transformation matrix into the boxSort algorithm so sign is kept?
        // Would be faster...
        payloadPtr->xyz = nodePayload->list[i].xyz*2.0f
            -glm::vec3(
                copysign(1.0f, nodePayload->list[i].xyz.x), 
                copysign(1.0f, nodePayload->list[i].xyz.y), 
                copysign(1.0f, nodePayload->list[i].xyz.z)
            );
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

void OctreeCapture::renderHelpLines(glm::mat4& view, glm::mat4& projection) {
    glm::mat4 model = glm::scale(glm::translate(glm::mat4x4(1.0f), this->capturePosition), glm::vec3(this->captureScale, this->captureScale, this->captureScale)) ;

    this->lineRenderer.setModelTransform(model);
    this->lineRenderer.render(view, projection);
}