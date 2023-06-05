#pragma once

#include <GL/glew.h>
//#include <librealsense2/rs.hpp>
#include <glm/mat4x4.hpp>

#include "shaders/pointcloudShader.h"

class PlyRenderer {
    GLuint vao;
    GLuint pointBuffer;
    GLuint colorBuffer;

    int pointCount = -1;

    PointcloudShader* shader;

public:
    PlyRenderer();
    ~PlyRenderer();

    void render(glm::mat4x4& model, glm::mat4x4& view, glm::mat4x4& proj);

    GLuint getPointBufferHandle() { return this->pointBuffer; }
    GLuint getColorBufferHandle() { return this->colorBuffer; }

    void uploadFrame(const glm::vec3* point, const glm::vec3* color, int count);
};
