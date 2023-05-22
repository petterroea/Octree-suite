#include <cudaHelpers.h>
#include "../players/octree/config.h"

void cudaRender(loadedOctreeType* in, int rootIdx, cudaSurfaceObject_t out, cudaSurfaceObject_t iterations, int imgWidth, int imgHeight, void* view, void* projection, int flipFlag);