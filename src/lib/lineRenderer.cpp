#include "lineRenderer.h"
#include "shader.h"

LineRendererShader::LineRendererShader() {
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
    gl_Position = projection*view*model*vec4(vertexPosition, 1.0f);
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

    this->shaderId = build_shader_program(shaders, 2);

    this->mat_model_location = glGetUniformLocation(this->shaderId, "model");
    this->mat_view_location = glGetUniformLocation(this->shaderId, "view");
    this->mat_projection_location = glGetUniformLocation(this->shaderId, "projection");
}

LineRendererShader::~LineRendererShader() {
    glDeleteShader(this->shaderId);
}

LineRenderer::LineRenderer(LineRendererShader* shader): shader(shader) {
    //VAO
    glGenVertexArrays(1, &this->vao);
    glBindVertexArray(this->vao);

    //Buffers
    glGenBuffers(1, &this->lineBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, this->lineBuffer);
    glVertexAttribPointer(
        0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
        3,                  // size
        GL_FLOAT,           // type
        GL_FALSE,           // normalized?
        0,                  // stride
        (void*)0            // array buffer offset
    );
    glEnableVertexAttribArray(0);

    glGenBuffers(1, &this->colorBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, this->colorBuffer);
    glVertexAttribPointer(
        1,                  
        3,                  // size
        GL_FLOAT,           // type
        GL_FALSE,           // normalized?
        0,                  // stride
        (void*)0            // array buffer offset
    );
    glEnableVertexAttribArray(1);

    //Cleanup
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

LineRenderer::~LineRenderer() {
    glDeleteVertexArrays(1, &this->vao);
    glDeleteBuffers(1, &this->lineBuffer);
    glDeleteBuffers(1, &this->colorBuffer);
}

void LineRenderer::reset() {
    this->lineSegments.clear();
    this->colors.clear();
}

void LineRenderer::drawLine(glm::vec3 p1, glm::vec3 p2, glm::vec3 color) {
    this->lineSegments.push_back(p1);
    this->lineSegments.push_back(p2);
    this->colors.push_back(color);
    this->colors.push_back(color);
}

void LineRenderer::setModelTransform(glm::mat4x4 mat) {
    this->modelTransform = mat;
}

void LineRenderer::render(glm::mat4& view, glm::mat4& proj) {
    // Upload this frame's line data
    // Dirty and inefficient but fast and easy
    glm::vec3* vertices = new glm::vec3[this->lineSegments.size()];
    glm::vec3* colors = new glm::vec3[this->colors.size()];

    int count = 0;
    for(auto itr = this->lineSegments.begin(); itr != this->lineSegments.end(); ++itr) {
        vertices[count++] = *itr;
    }
    count = 0;
    for(auto itr = this->colors.begin(); itr != this->colors.end(); ++itr) {
        colors[count++] = *itr;
    }

    glBindBuffer(GL_ARRAY_BUFFER, lineBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3)*this->lineSegments.size(), vertices, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3)*this->colors.size(), colors, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    delete[] vertices;
    delete[] colors;

    glUseProgram(this->shader->shaderId);

    // Update matrices
    glUniformMatrix4fv(this->shader->mat_model_location, 1, GL_FALSE, (const GLfloat*)&this->modelTransform);
    glUniformMatrix4fv(this->shader->mat_view_location, 1, GL_FALSE, (const GLfloat*)&view);
    glUniformMatrix4fv(this->shader->mat_projection_location, 1, GL_FALSE, (const GLfloat*)&proj);

    // Draw the lines
    glBindVertexArray(this->vao);
    glDrawArrays(GL_LINES, 0, this->lineSegments.size());

}
