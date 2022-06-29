#pragma once

#include <GL/glew.h>

class MeshShader {
    GLuint handle;

    GLuint mat_model_location;
    GLuint mat_view_location;
    GLuint mat_projection_location;

public:
    MeshShader();
    ~MeshShader();

    GLuint getHandle() { return this->handle; }

    GLuint getModelTransformUniformLocation() { return this->mat_model_location; }
    GLuint getViewTransformUniformLocation() { return this->mat_view_location; }
    GLuint getProjectionTraosformUniformLocation() { return this->mat_projection_location; }
};