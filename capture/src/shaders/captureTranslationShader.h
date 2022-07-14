#include <GL/glew.h>
// Since everything is on the GPU anyways we handle translation and matrix multiplication there
// Then we can copy it back to RAM
class CaptureTranslationShader {
    GLuint handle;

    GLuint mat_model_location;
    GLuint mat_view_location;
    GLuint mat_projection_location;
    GLuint textureLocation;

public:
    CaptureTranslationShader();
    ~CaptureTranslationShader();

    GLuint getHandle() { return this->handle; }

    GLuint getModelTransformUniformLocation() { return this->mat_model_location; }
    GLuint getViewTransformUniformLocation() { return this->mat_view_location; }
    GLuint getProjectionTraosformUniformLocation() { return this->mat_projection_location; }
    GLuint getTextureLocation() { return this->textureLocation; }
};