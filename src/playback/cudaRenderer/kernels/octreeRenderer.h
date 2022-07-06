#define CUDA_CATCH_ERROR {cudaError_t cuda_err_macro = cudaPeekAtLastError(); if(cuda_err_macro) { fprintf(stderr, "CUDA ERROR: %s at %d\n", cudaGetErrorString(cuda_err_macro), __LINE__); }}

void cudaRender(void* in, int rootOffset, cudaSurfaceObject_t out, int imgWidth, int imgHeight, void* view, void* projection, int flipFlag);