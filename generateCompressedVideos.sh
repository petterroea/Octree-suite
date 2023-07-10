#!/bin/bash
set -e

echo "Octree encoding benchmarker"

if [ -z "$1" ]; then
    echo "Please specify log dir"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Please specify data dir"
    exit 1
fi

if [ -z "$3" ]; then
    echo "Please specify frame dir"
    exit 1
fi

if [ -z "$4" ]; then
    echo "Please specify video dir"
    exit 1
fi

SOURCE="datasets/wave/oct"

mkdir -p $1
mkdir -p $2
mkdir -p $3
mkdir -p $4

function generate() {
    LOG_DIR=$1
    DATA_DIR=$2
    FRAME_DIR=$3
    VIDEO_DIR=$4
    FOLDER_NAME=$5

    if [ ! -f "${VIDEO_DIR}/${FOLDER_NAME}.mp4" ]; then
        mkdir "$3/${FOLDER_NAME}"
        bin/OctreeMasterVideoPlayer octree "$DATA_DIR/${FOLDER_NAME}" --record "${FRAME_DIR}/${FOLDER_NAME}/" --width 1280 --height 720
        echo "Generating video using ffmpeg"
        ffmpeg -framerate 30 -pattern_type glob -i "$FRAME_DIR/${FOLDER_NAME}/*.png" -c:v libx264 -crf 10 -pix_fmt yuv420p "${VIDEO_DIR}/${FOLDER_NAME}.mp4"
        echo "Starting next job..."
    fi
}

DCT_START_VALS=(1 2 4 8 16 32 64 128)
DCT_END_VALS=(1 4 8 16 32 64 128)

echo "starting"
for NAME in $(cat compressed_good.txt); do
    generate $1 $2 $3 $4 $NAME
done
