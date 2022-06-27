#include <GL/glew.h>

GLuint build_shader(GLuint type, const char* src);
GLuint build_shader_program(const GLuint* shaders, int shader_count);
