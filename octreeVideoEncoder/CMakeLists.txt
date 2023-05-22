set(CAPTURE_TO_PLY_BINARY_NAME "octreeVideoEncoder")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED OFF)

enable_language(CUDA)

find_package(CUDA REQUIRED)
message(STATUS "cuda_include=${CUDA_INCLUDE_DIRS}")

set(CUDA_FLAGS, "${CUDA_FLAGS} -g -G")

# Target
file(GLOB_RECURSE DIR_CAPTURE_TO_PLY_SRCS src/kernels/*.cu src/**/*.cpp src/*.cpp)
add_executable(${CAPTURE_TO_PLY_BINARY_NAME} ${DIR_CAPTURE_TO_PLY_SRCS})

target_compile_options(${CAPTURE_TO_PLY_BINARY_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -G
                       -g
                       >)


# Includes
target_include_directories(${CAPTURE_TO_PLY_BINARY_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(${CAPTURE_TO_PLY_BINARY_NAME} PUBLIC "../lib/src")
target_include_directories(${CAPTURE_TO_PLY_BINARY_NAME} PUBLIC "../rapidjson/include")

# Linking
target_link_libraries(${CAPTURE_TO_PLY_BINARY_NAME} ${CUDA_LIBRARIES})
target_link_libraries(${CAPTURE_TO_PLY_BINARY_NAME} OctreeMasterLib)