#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

void cudaTransformPointCloud(glm::vec3* pointSrc, cudaTextureObject_t colorSrc, glm::vec2* texCoords, glm::vec3* pointDst, glm::vec3* colorDst, int count, glm::mat4x4 transform);
