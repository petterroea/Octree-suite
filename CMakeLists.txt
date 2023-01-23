project(OctreeMaster)
cmake_minimum_required(VERSION 3.20)

# Have some standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Put output in a cleaner location
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Help vscode
#add_definitions(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)

# Options
option(BUILD_PLAYBACK "Build Playback program" ON)
option(BUILD_CAPTURE "Build the capture program" ON)
option(BUILD_PLY_TO_OCTREE "Build ply2octree" ON)
option(BUILD_CAPTURE_TO_PLY "Build ply2octree" ON)

include("./Imgui.cmake")

add_subdirectory("lib")
add_subdirectory("libPointcloudCapture")

if(BUILD_PLAYBACK)
    add_subdirectory("playback")
endif()
if(BUILD_CAPTURE)
    add_subdirectory("capture")
endif()
if(BUILD_PLY_TO_OCTREE)
    add_subdirectory("ply2octree")
endif()
if(BUILD_CAPTURE_TO_PLY)
    add_subdirectory("capture2ply")
endif()