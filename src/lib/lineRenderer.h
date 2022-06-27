#include <GL/glew.h>

#include <vector>

#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>

class LineRendererShader {
public:
    GLuint shaderId;

    GLuint mat_model_location;
    GLuint mat_view_location;
    GLuint mat_projection_location;

    LineRendererShader();
    ~LineRendererShader();
};

class LineRenderer {
private:
    LineRendererShader* shader;

    GLuint vao;

    GLuint lineBuffer;
    GLuint colorBuffer;

    glm::mat4x4 modelTransform;

    std::vector<glm::vec3> lineSegments;
    std::vector<glm::vec3> colors;
public:
    LineRenderer(LineRendererShader* shader);
    ~LineRenderer();

    void reset();
    void drawLine(glm::vec3 p1, glm::vec3 p2, glm::vec3 color);
    void setModelTransform(glm::mat4x4 mat);
    void render(glm::mat4& view, glm::mat4& proj);
};
