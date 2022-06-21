#include "pointcloudShader.h"
#include "shader.h"

PointcloudShader::PointcloudShader() {
    //Build shaders
    GLuint shaders[2] = {
        build_shader(GL_VERTEX_SHADER, R"(
#version 330 core
in vec3 vertexPosition;
in vec2 texcoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec2 texcoord_to_frag;
void main() {
    gl_Position = projection*view*model*vec4(vertexPosition, 1.0f);
    texcoord_to_frag = texcoord;
    //gl_Position.w = 0.0;
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

    this->mat_model_location = glGetUniformLocation(this->handle, "model");
    this->mat_view_location = glGetUniformLocation(this->handle, "view");
    this->mat_projection_location = glGetUniformLocation(this->handle, "projection");
    this->textureLocation = glGetUniformLocation(this->handle, "texture");
}

PointcloudShader::~PointcloudShader() {
    glDeleteShader(this->handle);
}