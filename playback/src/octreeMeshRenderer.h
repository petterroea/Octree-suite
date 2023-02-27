#include <vector>
#include <GL/glew.h>
#include <glm/mat4x4.hpp>

#include <octree/octree.h>
#include "shaders/meshShader.h"

class OctreeMeshRenderer {
    GLuint vao;
    GLuint vertexBuffer;
    GLuint colorBuffer;

    int vertexCount = 0;

    MeshShader shader;
    void pushOctreeCube(Octree<glm::vec3>* octree, std::vector<glm::vec3>& points, std::vector<glm::vec3>& colors, int level, int maxLevels, glm::vec3 offset);
public:
    OctreeMeshRenderer(Octree<glm::vec3>* octree);
    ~OctreeMeshRenderer();

    void render(glm::mat4 view, glm::mat4 projection);
};