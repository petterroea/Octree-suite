#include "json_helpers.h"

#include <stdexcept>

rapidjson::Value mat4x4_to_json_array(glm::mat4x4 matrix, rapidjson::Document::AllocatorType& allocator) {
    rapidjson::Value matrixArray;
    matrixArray.SetArray();
    matrixArray
        .PushBack(matrix[0][0], allocator).PushBack(matrix[0][1], allocator).PushBack(matrix[0][2], allocator).PushBack(matrix[0][3], allocator)
        .PushBack(matrix[1][0], allocator).PushBack(matrix[1][1], allocator).PushBack(matrix[1][2], allocator).PushBack(matrix[1][3], allocator)
        .PushBack(matrix[2][0], allocator).PushBack(matrix[2][1], allocator).PushBack(matrix[2][2], allocator).PushBack(matrix[2][3], allocator)
        .PushBack(matrix[3][0], allocator).PushBack(matrix[3][1], allocator).PushBack(matrix[3][2], allocator).PushBack(matrix[3][3], allocator);
    return matrixArray;
}

glm::mat4x4 json_array_to_mat4x4(rapidjson::Value& a) {
    if(a.Capacity() < 4*4) {
        throw std::invalid_argument("Invalid capacity");
    }
    glm::mat4x4 matrix(1.0f); 
    matrix[0][0] = a[0].GetFloat(); matrix[0][1] = a[1].GetFloat(); matrix[0][2] = a[2].GetFloat(); matrix[0][3] = a[3].GetFloat();
    matrix[1][0] = a[4].GetFloat(); matrix[1][1] = a[5].GetFloat(); matrix[1][2] = a[6].GetFloat(); matrix[1][3] = a[7].GetFloat();
    matrix[2][0] = a[8].GetFloat(); matrix[2][1] = a[9].GetFloat(); matrix[2][2] = a[10].GetFloat(); matrix[2][3] = a[11].GetFloat();
    matrix[3][0] = a[12].GetFloat(); matrix[3][1] = a[13].GetFloat(); matrix[3][2] = a[14].GetFloat(); matrix[3][3] = a[15].GetFloat();
    return matrix;
}