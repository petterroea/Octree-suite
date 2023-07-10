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

FLAGS="$5"

mkdir -p $1
mkdir -p $2
mkdir -p $3
mkdir -p $4

COLOR_IMPORTANCE_FACTORS=(0.1 0.01 0.001 0.0001 0.00001 0.000025 0.00005 0.000075 0.000001 0.000005)
NEARNESS_FACTORS=(9.999 9.9999 9.999925 9.99995 9.999975 9.99999 9.999999)
#COLOR_IMPORTANCE_FACTORS=(0.0001)
#NEARNESS_FACTORS=(9.999)

function generate() {
    LOG_DIR=$1
    DATA_DIR=$2
    FRAME_DIR=$3
    VIDEO_DIR=$4
    DCT_START=$5
    DCT_END=$6

    FOLDER_NAME="${DCT_START}_${DCT_END}"

    if [ "$FLAGS" != "noencode" ]; then
        echo "Testing $DCT_START $DCT_END"
        echo "test" | tee "${LOG_DIR}/${FOLDER_NAME}.log"
        bin/octreeVideoEncoder ${SOURCE} "${DATA_DIR}/${FOLDER_NAME}" --chunk_size 60 --quantization_start $DCT_START --quantization_end $DCT_END --skip_reduction true --compression_stats "${LOG_DIR}/${FOLDER_NAME}-numbers.csv" | tee "${LOG_DIR}/${FOLDER_NAME}.log"
        echo "Done encoding, rendering image"
    fi
    if [ "$FLAGS" != "novideo" ]; then
        if [ ! -f "${VIDEO_DIR}/${FOLDER_NAME}.mp4" ]; then
            mkdir "$3/${FOLDER_NAME}"
            bin/OctreeMasterVideoPlayer octree "$DATA_DIR/${FOLDER_NAME}" --record "${FRAME_DIR}/${FOLDER_NAME}/" --width 1280 --height 720
            echo "Generating video using ffmpeg"
            ffmpeg -framerate 30 -pattern_type glob -i "$FRAME_DIR/${FOLDER_NAME}/*.png" -c:v libx264 -crf 10 -pix_fmt yuv420p "${VIDEO_DIR}/${FOLDER_NAME}.mp4"
            echo "Starting next job..."
        fi
    fi
}

DCT_START_VALS=(1 2 4 8 16 32 64 128)
DCT_END_VALS=(1 4 8 16 32 64 128)

echo "starting"
for START in ${DCT_START_VALS[@]}; do
    for END in ${DCT_END_VALS[@]}; do
        generate $1 $2 $3 $4 $START $END
    done
done
