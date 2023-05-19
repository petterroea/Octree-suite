#pragma once

#include <GL/glew.h>

class PointcloudShader {
    GLuint handle;

    GLuint mat_model_location;
    GLuint mat_view_location;
    GLuint mat_projection_location;

public:
    PointcloudShader();
    ~PointcloudShader();

    GLuint getHandle() { return this->handle; }

    GLuint getModelTransformUniformLocation() { return this->mat_model_location; }
    GLuint getViewTransformUniformLocation() { return this->mat_view_location; }
    GLuint getProjectionTraosformUniformLocation() { return this->mat_projection_location; }
};