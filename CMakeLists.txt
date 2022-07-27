project(OctreeMaster)
cmake_minimum_required(VERSION 3.20)

# Have some standards
set(CMAKE_CXX_STANDARD 11)
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
option(BUILD_CONVERTER "Build ply2octree" ON)

include("./Imgui.cmake")
add_subdirectory("lib")
if(BUILD_PLAYBACK)
    add_subdirectory("playback")
endif()
if(BUILD_CAPTURE)
    add_subdirectory("capture")
endif()
if(BUILD_CONVERTER)
    add_subdirectory("ply2octree")
endif()