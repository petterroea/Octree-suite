cmake_minimum_required(VERSION 3.20)

project(OctreeMasterLib LANGUAGES CUDA CXX)
# Have some standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Put output in a cleaner location
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

# Help vscode
add_definitions(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)

# Library setup
#find_package(SDL2 REQUIRED)
#include_directories(${SDL2_INCLUDE_DIRS})
#message(STATUS, "sdl=${SDL2_INCLUDE_DIRS}")

enable_language(CUDA)

find_package(CUDA REQUIRED)
message(STATUS "cuda_include=${CUDA_INCLUDE_DIRS}")

find_package(OpenGL REQUIRED)

find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})
link_libraries(${GLEW_LIBRARIES})

file(GLOB_RECURSE DIR_LIB_SRC ${CMAKE_CURRENT_LIST_DIR}/src/kernels/*.cu ${CMAKE_CURRENT_LIST_DIR}/src/**/*.cpp ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
message(STATUS "lib_src=${DIR_LIB_SRC}")

add_library(${PROJECT_NAME} ${DIR_LIB_SRC})

target_include_directories(${PROJECT_NAME} PUBLIC "../rapidjson/include")
target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
# Define the library
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
#target_link_libraries(${CAPTURE_EXECUTABLE_NAME} ${SDL2_LIBRARIES})
#target_link_libraries(${CAPTURE_EXECUTABLE_NAME} GL)
#target_link_libraries(${CAPTURE_EXECUTABLE_NAME} ${realsense2_LIBRARY})
#target_link_libraries(${CAPTURE_EXECUTABLE_NAME} ${OpenCV_LIBS})
