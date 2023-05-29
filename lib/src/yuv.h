#pragma once

#include <glm/vec3.hpp>

struct yuv_image {
    unsigned char* y_image;
    unsigned char* u_image;
    unsigned char* v_image;
};

glm::vec3 rgb_to_yuv(const glm::vec3& rgb);
glm::vec3 yuv_to_rgb(const glm::vec3& yuv);

yuv_image image_to_yuv(unsigned char* rgb, int w, int h);

unsigned char* image_to_rgb(yuv_image* image, int w, int h);