# Realsense-multicam-capture

Uses OpenCV to determine camera positions relative to a ArUco board so you can capture a scene with multiple cameras.

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

The calibration is somewhat wrong due to the color sensor being offset from the depth sensor
