#pragma once
#include <vector>
#include <fstream>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>

#include "main.h"
#include "depthCamera.h"
#include "../lib/lineRenderer.h"
#include "../lib/octree.h"

#define ADDRESS_OCTREE_BY_VEC3(vec) ADDRESS_OCTREE(signbit(vec.x), signbit(vec.y), signbit(vec.z))

template <typename T>
struct SizedArray {
    T* list;
    int count;
    int max;
};

struct Point {
    glm::vec3 xyz;
    glm::vec3 rgb;
};

// Responsible for keeping capture configuration as well as performing the actual capture
class OctreeCapture {
    glm::vec3 capturePosition;
    float captureScale = 1.0f;

    LineRendererShader lineRendererShader;
    LineRenderer lineRenderer;

    void captureScene(std::vector<PointcloudCameraRendererPair*> cameras);
    void boxSort(Octree<SizedArray<Point>>* node, int level, int maxLevel);
    glm::vec3 serialize(Octree<SizedArray<Point>>* node, std::ofstream &treefile, int* writeHead, int* nodeLocation);
public:
    OctreeCapture();
    void displayGui(std::vector<PointcloudCameraRendererPair*> cameras);
    void renderHelpLines(glm::mat4& view, glm::mat4& projection);
};