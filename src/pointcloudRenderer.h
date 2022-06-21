#include <GL/glew.h>
#include <librealsense2/rs.hpp>
#include <glm/mat4x4.hpp>

#include "shaders/pointcloudShader.h"

class PointcloudRenderer {
    GLuint shaderId;
    GLuint vao;
    GLuint pointBuffer;
    GLuint texCoordBuffer;

    PointcloudShader* shader;

    int pointCount = 0;
public:
    PointcloudRenderer(PointcloudShader* shader);
    ~PointcloudRenderer();

    void render(glm::mat4x4& model, glm::mat4x4& view, glm::mat4x4& proj);
    void updateData(rs2::points& points);
};