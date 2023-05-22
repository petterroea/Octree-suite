set(CONVERTER_BINARY_NAME "ply2octree")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Target
file(GLOB_RECURSE DIR_CONVERTER_SRCS src/**/*.cpp src/*.cpp)
add_executable(${CONVERTER_BINARY_NAME} ${DIR_CONVERTER_SRCS})

# Includes
target_include_directories(${CONVERTER_BINARY_NAME} PUBLIC "../lib/src")

# Linking
target_link_libraries(${CONVERTER_BINARY_NAME} OctreeMasterLib)