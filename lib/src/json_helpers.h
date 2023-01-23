#include <rapidjson/document.h>
#include <glm/mat4x4.hpp>

rapidjson::Value mat4x4_to_json_array(glm::mat4x4 matrix, rapidjson::Document::AllocatorType& allocator);
glm::mat4x4 json_array_to_mat4x4(rapidjson::Value& array);