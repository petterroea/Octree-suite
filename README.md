# Octree suite

Tools for capturing 3d-scans, saving them as octrees, and rendering them

## Dependencies

 * Librealsense: https://github.com/IntelRealSense/librealsense
   - I had to build from scratch
 * SDL2
 * glew
 * glm
 * OpenCV

## Building

```
git submodule update --init --recursive
cmake .
make
```

## Caveats

OpenCV calibration is highly dependent on color camera resolution and ArUCo board size.
