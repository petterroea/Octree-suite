#!/bin/bash

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

#COLOR_IMPORTANCE_FACTORS=(0.0001)
#NEARNESS_FACTORS=(9.999)


#COLOR_IMPORTANCE_FACTORS=(0.1 0.01 0.001 0.0001 0.00001 0.000025 0.00005 0.000075 0.000001 0.000005)
#NEARNESS_FACTORS=(9.999 9.9999 9.999925 9.99995 9.999975 9.99999 9.999999)

NEARNESS_FACTORS=(0.9 0.95 0.99 0.995 0.999)

COLOR_IMPORTANCE_FACTORS=(0.01 0.02 0.03 0.05 0.07 0.09 0.1)

ARGS=()

echo "starting"
for COLOR_IMPORTANCE in ${COLOR_IMPORTANCE_FACTORS[@]}; do
    for NEARNESS_FACTOR in ${NEARNESS_FACTORS[@]}; do
        VALUE="$COLOR_IMPORTANCE;$NEARNESS_FACTOR"
        ARGS+=("$VALUE")
        #generate $1 $2 $3 $4 $COLOR_IMPORTANCE $NEARNESS_FACTOR
    done
done

echo ${ARGS[@]}

parallel -j 30 ./generateInner.sh $1 $2 $3 $4 {} $FLAGS ::: ${ARGS[@]}
echo "Done :)"