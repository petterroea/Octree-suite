#include "cudaBlitShader.h"
#include "../../lib/shader.h"

CudaBlitShader::CudaBlitShader() {
    //Build shaders
    GLuint shaders[2] = {
        build_shader(GL_VERTEX_SHADER, R"(
#version 330 core
in vec3 vertexPosition;
in vec2 texcoord;

out vec2 texcoord_to_frag;
void main() {
    gl_Position = vec4(vertexPosition, 1.0f);
    texcoord_to_frag = texcoord;
}
)"),
        build_shader(GL_FRAGMENT_SHADER, R"(
#version 330 core
out vec3 color;
in vec2 texcoord_to_frag;

uniform sampler2D texture;
void main() {
    color = texture2D(texture, texcoord_to_frag).rgb;
}
)")
    };

    this->handle = build_shader_program(shaders, 2);

    this->textureLocation = glGetUniformLocation(this->handle, "texture");
}

CudaBlitShader::~CudaBlitShader() {
    glDeleteShader(this->handle);
}