cmake_minimum_required(VERSION 3.20)

# set the project name
project(OctreeMaster LANGUAGES CUDA CXX)

set(CAPTURE_EXECUTABLE_NAME ${PROJECT_NAME}Capture)
set(PLAYBACK_EXECUTABLE_NAME ${PROJECT_NAME}Playback)

# Have some standards
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Put output in a cleaner location
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Help vscode
add_definitions(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)

# Library setup
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
message(STATUS, "sdl=${SDL2_INCLUDE_DIRS}")

find_package(OpenGL REQUIRED)

find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})
link_libraries(${GLEW_LIBRARIES})

find_package(realsense2 REQUIRED)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# Handle imgui(which doesn't support cmake)
set(IMGUI_SOURCE_DIR ${CMAKE_SOURCE_DIR}/imgui)
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
include_directories(${IMGUI_SOURCE_DIR} ${IMGUI_BACKENDS_DIR})

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

#aux_source_directory(src DIR_SRCS)

file(GLOB_RECURSE DIR_LIB_SRC src/lib/**/*.cpp src/lib/*.cpp)

# Define the capture executable
file(GLOB_RECURSE DIR_CAPTURE_SRCS src/capture/**/*.cpp src/capture/*.cpp)
add_executable(${CAPTURE_EXECUTABLE_NAME} ${DIR_CAPTURE_SRCS} ${DIR_LIB_SRC} ${IMGUI_SOURCE_FILES})
target_link_libraries(${CAPTURE_EXECUTABLE_NAME} ${SDL2_LIBRARIES})
target_link_libraries(${CAPTURE_EXECUTABLE_NAME} GL)
target_link_libraries(${CAPTURE_EXECUTABLE_NAME} ${realsense2_LIBRARY})
target_link_libraries(${CAPTURE_EXECUTABLE_NAME} ${OpenCV_LIBS})

# Define the playback executable
file(GLOB_RECURSE DIR_PLAYBACK_SRCS src/playback/cudaRenderer/kernels/*.cu src/playback/**/*.cpp src/playback/*.cpp)
add_executable(${PLAYBACK_EXECUTABLE_NAME} ${DIR_PLAYBACK_SRCS} ${DIR_LIB_SRC} ${IMGUI_SOURCE_FILES})
target_link_libraries(${PLAYBACK_EXECUTABLE_NAME} ${CUDA_LIBRARIES})
target_link_libraries(${PLAYBACK_EXECUTABLE_NAME} ${SDL2_LIBRARIES})
target_link_libraries(${PLAYBACK_EXECUTABLE_NAME} GL)
