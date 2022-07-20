#include "cudaPitchRgbToRgba.h"

#define THREADS_PER_BLOCK 32

/*
 * Takes a packed array of RGB pixels and copies them to an array of RGBA pixels, 
 * Setting the alpha value to 255
 * Doing this on GPU should be faster than CPU and also saves memory bandwidth.
 */
__global__ void kernel_pitchRgbToRgba(char* src, cudaSurfaceObject_t dst, int size, int width) {
    int pixelIdx = blockIdx.x*THREADS_PER_BLOCK+threadIdx.x;
    if(pixelIdx > size) {
        return;
    }
    unsigned char r = src[pixelIdx*3+0];
    unsigned char g = src[pixelIdx*3+1];
    unsigned char b = src[pixelIdx*3+2];
    unsigned int data = r | g << 8 | b << 16 | 0xFF000000;
    surf2Dwrite(data, dst, (pixelIdx%width)*sizeof(unsigned int), pixelIdx/width);
    //dst[pixelIdx] = *srcPtr | 0xFF000000;
}

void pitchRgbToRgba(void* src, cudaSurfaceObject_t dst, int size, int width) {
    int blockCount = (size+1)/THREADS_PER_BLOCK;
    kernel_pitchRgbToRgba<<<blockCount, THREADS_PER_BLOCK>>>((char*)src, dst, size, width);
}