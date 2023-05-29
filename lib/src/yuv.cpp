#include "yuv.h"

#include <algorithm>

glm::vec3 rgb_to_yuv(const glm::vec3& rgb) {
    return glm::vec3(
        0.257f * rgb.x + 0.504f * rgb.y + 0.098f * rgb.z +  16.0f,
        -0.148f * rgb.x - 0.291f * rgb.y + 0.439f * rgb.z + 128.0f,
        0.439f * rgb.x - 0.368f * rgb.y - 0.071f * rgb.z + 128.0f
    );
}

glm::vec3 yuv_to_rgb(const glm::vec3& yuv) {
    float y = yuv.x - 16.0f;
    float u = yuv.y - 128.0f;
    float v = yuv.z - 128.0f;

    return glm::vec3(
        1.164f * y              + 1.596f * v,
        1.164f * y - 0.392f * u - 0.813f * v,
        1.164f * y + 2.017f * u
    );
}

yuv_image image_to_yuv(unsigned char* rgb, int w, int h) {
    yuv_image image;

    image.y_image = new unsigned char[w*h];
    image.u_image = new unsigned char[w*h];
    image.v_image = new unsigned char[w*h];

    for(int x_pos = 0; x_pos < w; x_pos++) {
        for(int y_pos = 0; y_pos < h; y_pos++) {
            float r = static_cast<float>(rgb[(x_pos+y_pos*w)*3+0]);
            float g = static_cast<float>(rgb[(x_pos+y_pos*w)*3+1]);
            float b = static_cast<float>(rgb[(x_pos+y_pos*w)*3+2]);

            float y =  0.257f * r + 0.504f * g + 0.098f * b +  16.0f;
            float u = -0.148f * r - 0.291f * g + 0.439f * b + 128.0f;
            float v =  0.439f * r - 0.368f * g - 0.071f * b + 128.0f;

            image.y_image[x_pos+y_pos*w] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, y)));
            image.u_image[x_pos+y_pos*w] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, u)));
            image.v_image[x_pos+y_pos*w] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, v)));
        }
    }

    return image;
}

unsigned char* image_to_rgb(yuv_image* image, int w, int h) {
    unsigned char* rgb_image = new unsigned char[w*h*3];

    for(int x_pos = 0; x_pos < w; x_pos++) {
        for(int y_pos = 0; y_pos < h; y_pos++) {
            float y = static_cast<float>(image->y_image[x_pos + y_pos * w]) - 16.0f;
            float u = static_cast<float>(image->u_image[x_pos + y_pos * w]) - 128.0f;
            float v = static_cast<float>(image->v_image[x_pos + y_pos * w]) - 128.0f;

            float r = 1.164f * y              + 1.596f * v;
            float g = 1.164f * y - 0.392f * u - 0.813f * v;
            float b = 1.164f * y + 2.017f * u;

            rgb_image[(x_pos + y_pos * w) * 3 + 0 ] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, r)));
            rgb_image[(x_pos + y_pos * w) * 3 + 1 ] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, g)));
            rgb_image[(x_pos + y_pos * w) * 3 + 2 ] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, b)));
        }
    }

    return rgb_image;
}