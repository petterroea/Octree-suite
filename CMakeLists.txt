cmake_minimum_required(VERSION 3.20)

# set the project name
project(RealsenseDemo)

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
include_directories(RealsenseDemo ${SDL2_INCLUDE_DIRS})
message(STATUS, "sdl=${SDL2_INCLUDE_DIRS}")

find_package(OpenGL REQUIRED)

find_package(GLEW REQUIRED)
include_directories(RealsenseDemo ${GLEW_INCLUDE_DIRS})
link_libraries(${GLEW_LIBRARIES})

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
include_directories(RealsenseDemo ${IMGUI_SOURCE_DIR} ${IMGUI_BACKENDS_DIR})


aux_source_directory(src DIR_SRCS)
add_executable(RealsenseDemo ${DIR_SRCS} ${IMGUI_SOURCE_FILES})
target_link_libraries(RealsenseDemo ${SDL2_LIBRARIES})
target_link_libraries(RealsenseDemo GL)