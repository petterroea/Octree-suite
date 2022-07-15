set(PLAYBACK_BINARY_NAME "${PROJECT_NAME}Playback")

enable_language(CUDA)
# Library setup
find_package(SDL2 REQUIRED)
message(STATUS "sdl=${SDL2_INCLUDE_DIRS}")

find_package(OpenGL REQUIRED)

find_package(GLEW REQUIRED)
link_libraries(${GLEW_LIBRARIES})

find_package(CUDA REQUIRED)
message(STATUS "cuda_include=${CUDA_INCLUDE_DIRS}")

# Target
file(GLOB_RECURSE DIR_PLAYBACK_SRCS src/cudaRenderer/kernels/*.cu src/**/*.cpp src/*.cpp)
add_executable(${PLAYBACK_BINARY_NAME} ${DIR_PLAYBACK_SRCS} ${IMGUI_SOURCE_FILES})

# Includes
target_include_directories(${PLAYBACK_BINARY_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(${PLAYBACK_BINARY_NAME} PUBLIC ${SDL2_INCLUDE_DIRS})
target_include_directories(${PLAYBACK_BINARY_NAME} PUBLIC ${GLEW_INCLUDE_DIRS})
target_include_directories(${PLAYBACK_BINARY_NAME} PUBLIC ${IMGUI_SOURCE_DIR} ${IMGUI_BACKENDS_DIR})
target_include_directories(${PLAYBACK_BINARY_NAME} PUBLIC "../lib/src")

# Linking
target_link_libraries(${PLAYBACK_BINARY_NAME} ${CUDA_LIBRARIES})
target_link_libraries(${PLAYBACK_BINARY_NAME} ${SDL2_LIBRARIES})
target_link_libraries(${PLAYBACK_BINARY_NAME} GL)

target_link_libraries(${PLAYBACK_BINARY_NAME} OctreeMasterLib)