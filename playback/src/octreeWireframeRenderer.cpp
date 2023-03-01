#include "octreeWireframeRenderer.h"

#include <vector>
#include <iostream>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>


void OctreeWireframeRenderer::pushOctreeCube(PointerOctree<glm::vec3>* octree, std::vector<glm::vec3>& points, std::vector<glm::vec3>& colors, int level, int maxLevels, glm::vec3 offset) {
    float scale = pow(0.5f, static_cast<float>(level));
    // Always draw wireframes

    // z axis positive
    points.push_back((glm::vec3(1.0f, 1.0f, 1.0f)*scale)+offset);
    points.push_back((glm::vec3(-1.0f, 1.0f, 1.0f)*scale)+offset);

    points.push_back((glm::vec3(-1.0f, 1.0f, 1.0f)*scale)+offset);
    points.push_back((glm::vec3(-1.0f, -1.0f, 1.0f)*scale)+offset);

    points.push_back((glm::vec3(-1.0f, -1.0f, 1.0f)*scale)+offset);
    points.push_back((glm::vec3(1.0f, -1.0f, 1.0f)*scale)+offset);

    points.push_back((glm::vec3(1.0f, -1.0f, 1.0f)*scale)+offset);
    points.push_back((glm::vec3(1.0f, 1.0f, 1.0f)*scale)+offset);

    // Z axis negative
    points.push_back((glm::vec3(1.0f, 1.0f, -1.0f)*scale)+offset);
    points.push_back((glm::vec3(-1.0f, 1.0f, -1.0f)*scale)+offset);

    points.push_back((glm::vec3(-1.0f, 1.0f, -1.0f)*scale)+offset);
    points.push_back((glm::vec3(-1.0f, -1.0f, -1.0f)*scale)+offset);

    points.push_back((glm::vec3(-1.0f, -1.0f, -1.0f)*scale)+offset);
    points.push_back((glm::vec3(1.0f, -1.0f, -1.0f)*scale)+offset);

    points.push_back((glm::vec3(1.0f, -1.0f, -1.0f)*scale)+offset);
    points.push_back((glm::vec3(1.0f, 1.0f, -1.0f)*scale)+offset);

    // Connecting lines
    points.push_back((glm::vec3(1.0f, 1.0f, -1.0f)*scale)+offset);
    points.push_back((glm::vec3(1.0f, 1.0f, 1.0f)*scale)+offset);

    points.push_back((glm::vec3(-1.0f, 1.0f, -1.0f)*scale)+offset);
    points.push_back((glm::vec3(-1.0f, 1.0f, 1.0f)*scale)+offset);

    points.push_back((glm::vec3(-1.0f, -1.0f, -1.0f)*scale)+offset);
    points.push_back((glm::vec3(-1.0f, -1.0f, 1.0f)*scale)+offset);

    points.push_back((glm::vec3(1.0f, -1.0f, -1.0f)*scale)+offset);
    points.push_back((glm::vec3(1.0f, -1.0f, 1.0f)*scale)+offset);
    // Colors
    for(int i = 0; i < 2*4*3; i++) {
        colors.push_back(*octree->getPayload());
    }
    // If there is a child, iterate
    if(octree->getChildCount() != 0 && level != maxLevels) {
        //Iterate further
        for(int i = 0; i < 8; i++) {
            auto child = octree->getChildByIdx(i);
            if(child != nullptr) {
                float next_scale = pow(0.5f, static_cast<float>(level+1));
                // Sign bit set = negative
                glm::vec3 newOffset = offset + glm::vec3(
                    copysign(next_scale, (i & 0x1) == 1 ?-1.0f:1.0f),
                    copysign(next_scale, (i & 0x2) == 2 ?-1.0f:1.0f),
                    copysign(next_scale, (i & 0x4) == 4 ?-1.0f:1.0f) 
                );
                this->pushOctreeCube(child, points, colors, level+1, maxLevels, newOffset);
            }
        }
    }
}

OctreeWireframeRenderer::OctreeWireframeRenderer(PointerOctree<glm::vec3>* octree) {
    // Generate mesh
    std::vector<glm::vec3> points;
    std::vector<glm::vec3> colors;
    this->pushOctreeCube(octree, points, colors, 0, 20, glm::vec3(0.0f));

    this->vertexCount= points.size();
    if(points.size() != colors.size()) {
        std::cout << "Different color and point count" << std::endl;
        exit(1);
    }

    std::cout << "Building buffers" << std::endl;
    // Build VAO, buffers, and upload the data
    //VAO
    glGenVertexArrays(1, &this->vao);
    glBindVertexArray(this->vao);

    //Buffers
    glGenBuffers(1, &this->vertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, this->vertexBuffer);
    glVertexAttribPointer(
        0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
        3,                  // size
        GL_FLOAT,           // type
        GL_FALSE,           // normalized?
        0,                  // stride
        (void*)0            // array buffer offset
    );
    glEnableVertexAttribArray(0);

    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3)*points.size(), points.data(), GL_STATIC_DRAW);

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
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3)*colors.size(), colors.data(), GL_STATIC_DRAW);

    //Cleanup
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

OctreeWireframeRenderer::~OctreeWireframeRenderer() {
    glDeleteVertexArrays(1, &this->vao);
    glDeleteBuffers(1, &this->vertexBuffer);
    glDeleteBuffers(1, &this->vertexBuffer);
}

void OctreeWireframeRenderer::render(glm::mat4 view, glm::mat4 projection) {
    glUseProgram(this->shader.getHandle());

    // Transpose matrixes since GLM matrixes are column major
    glm::mat4x4 model(1.0f);
    glm::mat4x4 model_transposed = glm::transpose(model);
    glm::mat4x4 view_transposed = glm::transpose(view);
    glm::mat4x4 projection_transposed = glm::transpose(projection);

    glUniformMatrix4fv(this->shader.getModelTransformUniformLocation(), 1, GL_FALSE, (const GLfloat*) &model_transposed);
    glUniformMatrix4fv(this->shader.getViewTransformUniformLocation(), 1, GL_FALSE, (const GLfloat*) &view_transposed);
    glUniformMatrix4fv(this->shader.getProjectionTraosformUniformLocation(), 1, GL_FALSE, (const GLfloat*) &projection_transposed);

    glBindVertexArray(this->vao);
    glDrawArrays(GL_LINES, 0, vertexCount);
    glUseProgram(0);
}