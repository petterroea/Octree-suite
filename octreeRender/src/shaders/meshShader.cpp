#include "meshShader.h"
#include <shader.h>

MeshShader::MeshShader() {
    //Build shaders
    GLuint shaders[2] = {
        build_shader(GL_VERTEX_SHADER, R"(
#version 330 core
in vec3 vertexPosition;
in vec3 color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 color_to_frag;
void main() {
    gl_Position = vec4(vertexPosition, 1.0f)*model*view*projection;
    color_to_frag = color;
    //gl_Position.w = 0.0;
}
)"),
        build_shader(GL_FRAGMENT_SHADER, R"(
#version 330 core
out vec3 color;
in vec3 color_to_frag;

void main() {
    color = color_to_frag;
}
)")
    };

    this->handle = build_shader_program(shaders, 2);

    this->mat_model_location = glGetUniformLocation(this->handle, "model");
    this->mat_view_location = glGetUniformLocation(this->handle, "view");
    this->mat_projection_location = glGetUniformLocation(this->handle, "projection");
}

MeshShader::~MeshShader() {
    glDeleteShader(this->handle);
}