#include <cudaHelpers.h>

void cudaRender(void* in, int rootOffset, cudaSurfaceObject_t out, cudaSurfaceObject_t iterations, int imgWidth, int imgHeight, void* view, void* projection, int flipFlag);