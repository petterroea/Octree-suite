cmake_minimum_required(VERSION 3.20)

set(LIB_POINTCLOUDCAPTURE_NAME "LibPointcloudCapture")
project(${LIB_POINTCLOUDCATURE_NAME} LANGUAGES CUDA CXX)

# Have some standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Put output in a cleaner location
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

# Help vscode
add_definitions(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)

# Library setup
find_package(SDL2 REQUIRED)
#message(STATUS, "sdl=${SDL2_INCLUDE_DIRS}")

find_package(OpenGL REQUIRED)

find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})

find_package(CUDA REQUIRED)

file(GLOB_RECURSE DIR_LIBPOINTCLOUD_SRC src/kernels/*.cu ${CMAKE_CURRENT_LIST_DIR}/src/**/*.cpp ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
message(STATUS "lib_src=${DIR_LIBPOINTCLOUD_SRC}")
message(STATUS "lib_name=${LIB_POINTCLOUDCAPTURE_NAME}")

add_library(${LIB_POINTCLOUDCAPTURE_NAME} STATIC ${DIR_LIBPOINTCLOUD_SRC})
add_library(${LIB_POINTCLOUDCAPTURE_NAME}_GL STATIC ${DIR_LIBPOINTCLOUD_SRC} ${IMGUI_SOURCE_FILES})

target_compile_options(${LIB_POINTCLOUDCAPTURE_NAME} PUBLIC "-DHEADLESS_RELEASE=1")

target_include_directories(${LIB_POINTCLOUDCAPTURE_NAME} PUBLIC "../rapidjson/include")
target_include_directories(${LIB_POINTCLOUDCAPTURE_NAME}_GL PUBLIC "../rapidjson/include")

target_include_directories(${LIB_POINTCLOUDCAPTURE_NAME} PUBLIC "../lib/src")
target_include_directories(${LIB_POINTCLOUDCAPTURE_NAME}_GL PUBLIC "../lib/src")

target_include_directories(${LIB_POINTCLOUDCAPTURE_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(${LIB_POINTCLOUDCAPTURE_NAME}_GL PUBLIC ${CUDA_INCLUDE_DIRS})

target_include_directories(${LIB_POINTCLOUDCAPTURE_NAME}_GL PUBLIC ${IMGUI_SOURCE_DIR} ${IMGUI_BACKENDS_DIR})

target_include_directories(${LIB_POINTCLOUDCAPTURE_NAME}_GL PUBLIC ${SDL2_INCLUDE_DIRS})

# Define the library
target_link_libraries(${LIB_POINTCLOUDCAPTURE_NAME} ${CUDA_LIBRARIES})
target_link_libraries(${LIB_POINTCLOUDCAPTURE_NAME} ${realsense2_LIBRARY})
target_link_libraries(${LIB_POINTCLOUDCAPTURE_NAME} OctreeMasterLib)

target_link_libraries(${LIB_POINTCLOUDCAPTURE_NAME}_GL ${CUDA_LIBRARIES})
target_link_libraries(${LIB_POINTCLOUDCAPTURE_NAME}_GL ${realsense2_LIBRARY})
target_link_libraries(${LIB_POINTCLOUDCAPTURE_NAME}_GL GL)
target_link_libraries(${LIB_POINTCLOUDCAPTURE_NAME}_GL OctreeMasterLib)
target_link_libraries(${LIB_POINTCLOUDCAPTURE_NAME}_GL ${GLEW_LIBRARIES})