# Octree suite

Tools for capturing 3d-scans, saving them as octrees, and rendering them. These were written as part of my master's thesis. I don't think you will find anything here useufl, but if you do, go ahead and use it.

The repo contains an encoder that encodes sequences of octrees as a single file where similar parts of the octrees are joined together. We can also compress color using DCT, allbeit in a primitve fashion.

Datasets for testing are available on demand, but you probably don't want them.

## Building

```
git submodule update --init --recursive
cmake .
make
```
## Project overview

### capture

Contains software that captures pointclouds

### playback

Contains the source code for executables relating to playing back recorded 3d captures

### capture2ply

Converts realsense capture rosbag files to ply sequences

### videoPlayer

Plays octree video files

### octreeVideoEncoder

Encodes octree video files.