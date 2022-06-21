#include <iostream>
#include "pointcloudRenderer.h"

PointcloudRenderer::PointcloudRenderer(PointcloudShader* shader): shader(shader) {
    //VAO
    glGenVertexArrays(1, &this->vao);
    glBindVertexArray(this->vao);

    //Buffers
    glGenBuffers(1, &this->pointBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, this->pointBuffer);
    glVertexAttribPointer(
        0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
        3,                  // size
        GL_FLOAT,           // type
        GL_FALSE,           // normalized?
        0,                  // stride
        (void*)0            // array buffer offset
    );
    glEnableVertexAttribArray(0);

    glGenBuffers(1, &this->texCoordBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, this->texCoordBuffer);
    glVertexAttribPointer(
        1,                  
        2,                  // size
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

PointcloudRenderer::~PointcloudRenderer() {
    glDeleteVertexArrays(1, &this->vao);
    glDeleteBuffers(1, &this->pointBuffer);
    glDeleteBuffers(1, &this->texCoordBuffer);
}

void PointcloudRenderer::render(glm::mat4x4& model, glm::mat4x4& view, glm::mat4x4& proj) {
    glUseProgram(this->shader->getHandle());

    glUniformMatrix4fv(this->shader->getModelTransformUniformLocation(), 1, GL_FALSE, (const GLfloat*) &model);
    glUniformMatrix4fv(this->shader->getViewTransformUniformLocation(), 1, GL_FALSE, (const GLfloat*) &view);
    glUniformMatrix4fv(this->shader->getProjectionTraosformUniformLocation(), 1, GL_FALSE, (const GLfloat*) &proj);
    glUniform1i(this->shader->getTextureLocation(), 0);
    
    glBindVertexArray(this->vao);
    glDrawArrays(GL_POINTS, 0, this->pointCount);
    glUseProgram(0);
}


void PointcloudRenderer::updateData(rs2::points& points) {
    glBindBuffer(GL_ARRAY_BUFFER, this->pointBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(rs2::vertex)*points.size(), points.get_vertices(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, this->texCoordBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(rs2::texture_coordinate)*points.size(), points.get_texture_coordinates(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    this->pointCount = points.size();
}