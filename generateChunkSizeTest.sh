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

LOG_DIR=$1
DATA_DIR=$2
FRAME_DIR=$3
VIDEO_DIR=$4

CHUNK_SIZE=$5

DCT_START=1
DCT_END=1

parallel -j 3 bin/octreeVideoEncoder ${SOURCE} "${DATA_DIR}/{}" --chunk_size {} --quantization_start $DCT_START --quantization_end $DCT_END --skip_reduction true --compression_stats "${LOG_DIR}/{}-numbers.csv" ::: {1..75}