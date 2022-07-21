# Capture

Software for recording from multiple RealSense cameras at the same time, transforming the captures to the world-space, and saving it to file.

Also handles trimming of points outside a specified capture bounds

## Requirements

 * Librealsense: https://github.com/IntelRealSense/librealsense
   - I had to build from scratch
 * OpenCV
 * CUDA
 * SDL2
 * glew
 * glm


## Caveats

OpenCV calibration is highly dependent on color camera resolution and ArUCo board size.