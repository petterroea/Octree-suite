set(OCTREE_VIDEO_PLAYER_BINARY_NAME "${PROJECT_NAME}VideoPlayer")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED OFF)

enable_language(CUDA)
# Library setup
find_package(SDL2 REQUIRED)
message(STATUS "sdl=${SDL2_INCLUDE_DIRS}")

find_package(OpenGL REQUIRED)

find_package(GLEW REQUIRED)
link_libraries(${GLEW_LIBRARIES})

find_package(CUDA REQUIRED)
message(STATUS "cuda_include=${CUDA_INCLUDE_DIRS}")

# ZLIB
find_package(ZLIB)

# Target
file(GLOB_RECURSE DIR_OCTREE_VIDEO_PLAYER_SRCS src/kernels/*.cu src/**/*.cpp src/*.cpp)
add_executable(${OCTREE_VIDEO_PLAYER_BINARY_NAME} ${DIR_OCTREE_VIDEO_PLAYER_SRCS} ${IMGUI_SOURCE_FILES})

target_compile_options(${OCTREE_VIDEO_PLAYER_BINARY_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -G
                       -g
                       >)

# Includes
target_include_directories(${OCTREE_VIDEO_PLAYER_BINARY_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(${OCTREE_VIDEO_PLAYER_BINARY_NAME} PUBLIC ${SDL2_INCLUDE_DIRS})
target_include_directories(${OCTREE_VIDEO_PLAYER_BINARY_NAME} PUBLIC ${GLEW_INCLUDE_DIRS})
target_include_directories(${OCTREE_VIDEO_PLAYER_BINARY_NAME} PUBLIC ${IMGUI_SOURCE_DIR} ${IMGUI_BACKENDS_DIR})
target_include_directories(${OCTREE_VIDEO_PLAYER_BINARY_NAME} PUBLIC "../lib/src")
target_include_directories(${OCTREE_VIDEO_PLAYER_BINARY_NAME} PUBLIC "../stb")
target_include_directories(${OCTREE_VIDEO_PLAYER_BINARY_NAME} PUBLIC "../rapidjson/include")

# Linking
target_link_libraries(${OCTREE_VIDEO_PLAYER_BINARY_NAME} ${CUDA_LIBRARIES})
target_link_libraries(${OCTREE_VIDEO_PLAYER_BINARY_NAME} ${SDL2_LIBRARIES})
target_link_libraries(${OCTREE_VIDEO_PLAYER_BINARY_NAME} GL)

target_link_libraries(${OCTREE_VIDEO_PLAYER_BINARY_NAME} OctreeMasterLib)
target_link_libraries(${OCTREE_VIDEO_PLAYER_BINARY_NAME} ZLIB::ZLIB)