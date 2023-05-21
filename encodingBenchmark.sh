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

mkdir -p $1
mkdir -p $2
mkdir -p $3
mkdir -p $4

COLOR_IMPORTANCE_FACTORS=(0.1 0.01 0.001 0.0001 0.00001 0.000025 0.00005 0.000075 0.000001 0.000005)
NEARNESS_FACTORS=(9.999 9.9999 9.999925 9.99995 9.999975 9.99999 9.999999)
#COLOR_IMPORTANCE_FACTORS=(0.0001)
#NEARNESS_FACTORS=(9.999)

for COLOR_FACTOR in ${COLOR_IMPORTANCE_FACTORS[@]}; do
    for NEARNESS_FACTOR in ${NEARNESS_FACTORS[@]}; do
        echo "Testing $COLOR_FACTOR $NEARNESS_FACTOR"
        echo "test" | tee "$1/color_${COLOR_FACTOR}_nearness_${NEARNESS_FACTOR}.log"
        bin/octreeVideoEncoder ~/wave "$2/c_${COLOR_FACTOR}_n_${NEARNESS_FACTOR}" --limit 40 --chunk_size 20 --encoding_threads 25 --chunk_concurrency 1 | tee "$1/color_${COLOR_FACTOR}_nearness_${NEARNESS_FACTOR}.log"
        echo "Done encoding, rendering image"
	mkdir "$3/c_${COLOR_FACTOR}_n_${NEARNESS_FACTOR}"
        bin/OctreeMasterVideoPlayer octree "$2/c_${COLOR_FACTOR}_n_${NEARNESS_FACTOR}" --record "$3/c_${COLOR_FACTOR}_n_${NEARNESS_FACTOR}/" --width 1280 --height 720
        echo "Generating video using ffmpeg"
        ffmpeg -framerate 30 -pattern_type glob -i "$3/c_${COLOR_FACTOR}_n_${NEARNESS_FACTOR}/*.png" -c:v libx264 -crf 10 -pix_fmt yuv420p "$4/c_${COLOR_FACTOR}_n_${NEARNESS_FACTOR}.mp4"
        echo "Starting next job..."
    done
done

