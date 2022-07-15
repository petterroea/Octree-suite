set(CAPTURE_BINARY_NAME "${PROJECT_NAME}Capture")

cmake_minimum_required(VERSION 3.20)

# Library setup
find_package(SDL2 REQUIRED)
message(STATUS "sdl=${SDL2_INCLUDE_DIRS}")

find_package(OpenGL REQUIRED)

find_package(GLEW REQUIRED)
link_libraries(${GLEW_LIBRARIES})

find_package(realsense2 REQUIRED)

find_package(OpenCV REQUIRED)

# Target
file(GLOB_RECURSE DIR_CAPTURE_SRCS src/**/*.cpp src/*.cpp)
add_executable(${CAPTURE_BINARY_NAME} ${DIR_CAPTURE_SRCS} ${IMGUI_SOURCE_FILES})

# Includes
target_include_directories(${CAPTURE_BINARY_NAME} PUBLIC ${SDL2_INCLUDE_DIRS})
target_include_directories(${CAPTURE_BINARY_NAME} PUBLIC ${GLEW_INCLUDE_DIRS})
target_include_directories(${CAPTURE_BINARY_NAME} PUBLIC ${IMGUI_SOURCE_DIR} ${IMGUI_BACKENDS_DIR})
target_include_directories(${CAPTURE_BINARY_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_include_directories(${CAPTURE_BINARY_NAME} PUBLIC "../lib/src")

# Linking
target_link_libraries(${CAPTURE_BINARY_NAME} ${SDL2_LIBRARIES})
target_link_libraries(${CAPTURE_BINARY_NAME} GL)
target_link_libraries(${CAPTURE_BINARY_NAME} ${realsense2_LIBRARY})
target_link_libraries(${CAPTURE_BINARY_NAME} ${OpenCV_LIBS})
target_link_libraries(${CAPTURE_BINARY_NAME} OctreeMasterLib)
