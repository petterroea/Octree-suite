set(CAPTURE_TO_PLY_BINARY_NAME "octreeVideoEncoder")

enable_language(CUDA)

find_package(CUDA REQUIRED)
message(STATUS "cuda_include=${CUDA_INCLUDE_DIRS}")

# Target
file(GLOB_RECURSE DIR_CAPTURE_TO_PLY_SRCS src/kernels/*.cu src/**/*.cpp src/*.cpp)
add_executable(${CAPTURE_TO_PLY_BINARY_NAME} ${DIR_CAPTURE_TO_PLY_SRCS})

# Includes
target_include_directories(${CAPTURE_TO_PLY_BINARY_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(${CAPTURE_TO_PLY_BINARY_NAME} PUBLIC "../lib/src")
target_include_directories(${CAPTURE_TO_PLY_BINARY_NAME} PUBLIC "../rapidjson/include")

# Linking
target_link_libraries(${CAPTURE_TO_PLY_BINARY_NAME} ${CUDA_LIBRARIES})
target_link_libraries(${CAPTURE_TO_PLY_BINARY_NAME} OctreeMasterLib)