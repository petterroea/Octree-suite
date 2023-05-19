#include <iostream>
#include <exception>

#include "plyRenderer.h"

PlyRenderer::PlyRenderer() {
    this->shader = new PointcloudShader();

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

PlyRenderer::~PlyRenderer() {
    glDeleteVertexArrays(1, &this->vao);
    glDeleteBuffers(1, &this->pointBuffer);
    glDeleteBuffers(1, &this->colorBuffer);
}

void PlyRenderer::render(glm::mat4x4& model, glm::mat4x4& view, glm::mat4x4& projection) {
    if(this->pointCount == -1) {
        throw std::invalid_argument("Tried to render before points have been uploaded");
    }
    glUseProgram(this->shader->getHandle());

    // Transpose matrixes since GLM matrixes are column major
    glm::mat4x4 model_transposed = glm::transpose(model);
    glm::mat4x4 view_transposed = glm::transpose(view);
    glm::mat4x4 projection_transposed = glm::transpose(projection);

    glUniformMatrix4fv(this->shader->getModelTransformUniformLocation(), 1, GL_FALSE, (const GLfloat*) &model_transposed);
    glUniformMatrix4fv(this->shader->getViewTransformUniformLocation(), 1, GL_FALSE, (const GLfloat*) &view_transposed);
    glUniformMatrix4fv(this->shader->getProjectionTraosformUniformLocation(), 1, GL_FALSE, (const GLfloat*) &projection_transposed);

    glBindVertexArray(this->vao);
    std::cout << "Rendering " << this->pointCount << " points" << std::endl;
    glDrawArrays(GL_POINTS, 0, this->pointCount);
    glUseProgram(0);
}

void PlyRenderer::uploadFrame(const glm::vec3* point, const glm::vec3* color, int count) {
    std::cout << "Uploading frame with " << count << " points." << std::endl;
    glBindBuffer(GL_ARRAY_BUFFER, this->pointBuffer);
    glBufferData(GL_ARRAY_BUFFER, count*sizeof(glm::vec3), point, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, this->colorBuffer);
    glBufferData(GL_ARRAY_BUFFER, count*sizeof(glm::vec3), color, GL_DYNAMIC_DRAW);
    this->pointCount = count;
}