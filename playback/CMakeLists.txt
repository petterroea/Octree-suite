cmake_minimum_required(VERSION 3.20)

include("../lib/CMakeLists.txt")

# set the project name
project(OctreeMasterPlayback)

# Have some standards
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Put output in a cleaner location
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

# Help vscode
add_definitions(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)


# Library setup
find_package(SDL2 REQUIRED)
message(STATUS, "sdl=${SDL2_INCLUDE_DIRS}")

find_package(OpenGL REQUIRED)

find_package(GLEW REQUIRED)
link_libraries(${GLEW_LIBRARIES})

# Handle imgui(which doesn't support cmake)
set(IMGUI_SOURCE_DIR ${CMAKE_SOURCE_DIR}/../imgui)
set(IMGUI_BACKENDS_DIR ${IMGUI_SOURCE_DIR}/backends)

set(IMGUI_SOURCE_FILES
    ${IMGUI_SOURCE_DIR}/imgui_demo.cpp
    ${IMGUI_SOURCE_DIR}/imgui_draw.cpp
    ${IMGUI_SOURCE_DIR}/imgui_tables.cpp
    ${IMGUI_SOURCE_DIR}/imgui.cpp
    ${IMGUI_SOURCE_DIR}/imgui_widgets.cpp
    ${IMGUI_BACKENDS_DIR}/imgui_impl_opengl3.cpp
    ${IMGUI_BACKENDS_DIR}/imgui_impl_sdl.cpp
)

find_package(CUDA REQUIRED)
message(STATUS, "cuda_include=${CUDA_INCLUDE_DIRS}")

#aux_source_directory(src DIR_SRCS)

# Target
file(GLOB_RECURSE DIR_PLAYBACK_SRCS src/cudaRenderer/kernels/*.cu src/**/*.cpp src/*.cpp)
add_executable(${PROJECT_NAME} ${DIR_PLAYBACK_SRCS} ${IMGUI_SOURCE_FILES})

# Includes
target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${SDL2_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${GLEW_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${IMGUI_SOURCE_DIR} ${IMGUI_BACKENDS_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC "../lib/src")

# Linking
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${SDL2_LIBRARIES})
target_link_libraries(${PROJECT_NAME} GL)

target_link_libraries(${PROJECT_NAME} OctreeMasterLib)