#include <vector>
#include <GL/glew.h>
#include <glm/mat4x4.hpp>

#include <octree/octree.h>
#include "shaders/meshShader.h"

class OctreeWireframeRenderer {
    GLuint vao;
    GLuint vertexBuffer;
    GLuint colorBuffer;

    int vertexCount = 0;

    MeshShader shader;
    void pushOctreeCube(Octree<glm::vec3>* octree, std::vector<glm::vec3>& points, std::vector<glm::vec3>& colors, int level, int maxLevels, glm::vec3 offset);
public:
    OctreeWireframeRenderer(Octree<glm::vec3>* octree);
    ~OctreeWireframeRenderer();

    void render(glm::mat4 view, glm::mat4 projection);
};