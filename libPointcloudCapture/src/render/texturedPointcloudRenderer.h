#pragma once

#include <GL/glew.h>
#include <librealsense2/rs.hpp>
#include <glm/mat4x4.hpp>

#include "../depthCamera/types.h"
#include "shaders/pointcloudShader.h"

class TexturedPointcloudRenderer {
    GLuint vao;
    GLuint pointBuffer;
    GLuint texCoordBuffer;

    GLuint colorTexture;
    // Only used for debug
    GLuint depthTexture;

    PointcloudShader* shader;
    VideoMode mode;

    GLuint createTexture(GLuint format, GLuint type, int width, int height);
public:
    TexturedPointcloudRenderer(VideoMode mode, PointcloudShader* shader);
    ~TexturedPointcloudRenderer();

    void render(glm::mat4x4& model, glm::mat4x4& view, glm::mat4x4& proj, int pointCount);

    GLuint getColorTextureHandle() { return this->colorTexture; }
    GLuint getDepthTextureHandle() { return this->depthTexture; }
    GLuint getPointBufferHandle() { return this->pointBuffer; }
    GLuint getTextureCoordBufferHandle() { return this->texCoordBuffer; }
};