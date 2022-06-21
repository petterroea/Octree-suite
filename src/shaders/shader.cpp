#include "shader.h"

#include <GL/glu.h>
#include <SDL2/SDL_opengl.h>

#include <cstring>
#include <iostream>

GLuint build_shader(GLuint type, const char* src) {
    GLuint id = 0;
    id = glCreateShader(type);
    std::cout << "created shader id " << id << std::endl;

    glShaderSource(id, 1, &src, NULL);
    glCompileShader(id);
    GLint compiled = GL_FALSE;
    glGetShaderiv(id, GL_COMPILE_STATUS, &compiled);

    if(compiled != GL_TRUE) {
        GLint length = 0;
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
        GLchar* error_log = new char[length];
        glGetShaderInfoLog(id, length, &length, error_log);

        std::cout << "Failed to compile" << error_log << std::endl;
        delete error_log;
        return 0;
    }
    std::cout << "built shader" << std::endl;
    return id;
}

GLuint build_shader_program(const GLuint* shaders, int shader_count) {
    GLuint id = glCreateProgram();
    for(int i = 0; i < shader_count; i++) {
        glAttachShader(id, shaders[i]);
    }
    glLinkProgram(id);

    GLint success = GL_FALSE;
    glGetProgramiv(id, GL_LINK_STATUS, &success);
    if(success != GL_TRUE) {
        std::cout << "Failed to link" << std::endl;
        return 0;
    }
    return id;
}