#include <iostream>
#include "texturedPointcloudRenderer.h"

TexturedPointcloudRenderer::TexturedPointcloudRenderer(VideoMode mode, PointcloudShader* shader): mode(mode), shader(shader) {
    //VAO
    glGenVertexArrays(1, &this->vao);
    glBindVertexArray(this->vao);

    //Cuda complains if we try to bind it to OpenGl before the buffer contains data
    int expectedMaxParticles = mode.colorWidth*mode.colorHeight;
    char* placeholder = new char[expectedMaxParticles*sizeof(glm::vec3)];
    //Buffers
    glGenBuffers(1, &this->pointBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, this->pointBuffer);
    glBufferData(GL_ARRAY_BUFFER, expectedMaxParticles*sizeof(glm::vec3), placeholder, GL_DYNAMIC_DRAW);
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
    glBufferData(GL_ARRAY_BUFFER, expectedMaxParticles*sizeof(glm::vec2), placeholder, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(
        1,                  
        2,                  // size
        GL_FLOAT,           // type
        GL_FALSE,           // normalized?
        0,                  // stride
        (void*)0            // array buffer offset
    );
    glEnableVertexAttribArray(1);
    delete[] placeholder;

    //Cleanup
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    this->colorTexture = this->createTexture(GL_RGB, GL_UNSIGNED_BYTE, mode.colorWidth, mode.colorHeight);
    this->depthTexture = this->createTexture(GL_RED, GL_UNSIGNED_SHORT, mode.depthWidth, mode.depthHeight);

    std::cout << "Color texture " << this->colorTexture << std::endl;
    std::cout << "Depth texture " << this->depthTexture << std::endl;
}

TexturedPointcloudRenderer::~TexturedPointcloudRenderer() {
    glDeleteVertexArrays(1, &this->vao);
    glDeleteBuffers(1, &this->pointBuffer);
    glDeleteBuffers(1, &this->texCoordBuffer);

    glDeleteTextures(1, &this->colorTexture);
    glDeleteTextures(1, &this->depthTexture);
}

void TexturedPointcloudRenderer::render(glm::mat4x4& model, glm::mat4x4& view, glm::mat4x4& projection, int pointCount) {
    glUseProgram(this->shader->getHandle());

    // Transpose matrixes since GLM matrixes are column major
    glm::mat4x4 model_transposed = glm::transpose(model);
    glm::mat4x4 view_transposed = glm::transpose(view);
    glm::mat4x4 projection_transposed = glm::transpose(projection);

    glUniformMatrix4fv(this->shader->getModelTransformUniformLocation(), 1, GL_FALSE, (const GLfloat*) &model_transposed);
    glUniformMatrix4fv(this->shader->getViewTransformUniformLocation(), 1, GL_FALSE, (const GLfloat*) &view_transposed);
    glUniformMatrix4fv(this->shader->getProjectionTraosformUniformLocation(), 1, GL_FALSE, (const GLfloat*) &projection_transposed);
    glUniform1i(this->shader->getTextureLocation(), 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, this->colorTexture);
    
    glBindVertexArray(this->vao);
    glDrawArrays(GL_POINTS, 0, pointCount);
    glUseProgram(0);
}

GLuint TexturedPointcloudRenderer::createTexture(GLuint format, GLuint type, int width, int height) {
    // Create a OpenGL texture identifier
    GLuint image_texture;
    glGenTextures(1, &image_texture);
    std::cout << "Got " << image_texture << std::endl;
    glBindTexture(GL_TEXTURE_2D, image_texture);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same

    // Upload pixels into texture
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, type, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    return image_texture;
}
