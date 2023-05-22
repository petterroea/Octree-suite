#pragma once

#include <GL/glew.h>

class CudaBlitShader {
    GLuint handle;

    GLuint textureLocation;

public:
    CudaBlitShader();
    ~CudaBlitShader();

    GLuint getHandle() { return this->handle; }

    GLuint getTextureLocation() { return this->textureLocation; }
};