SOURCE="datasets/wave/oct"

FLAGS="$6"

function try_and_fail() {
    SOURCE=$1
    DIR=$2
    STATS_FILE=$3
    LOG_DIR=$4
    COLOR_IMPORTANCE=$5
    NEARNESS_FACTOR=$6

    bin/octreeVideoEncoder $SOURCE $DIR --chunk_size 10 --compression_stats $STATS_FILE --encoding_threads 5 --chunk_concurrency 7 --nearness_factor $NEARNESS_FACTOR --color_importance $COLOR_IMPORTANCE --quantization_start 1 --quantization_end 1 | tee $LOG_DIR
}

function generate() {
    LOG_DIR=$1
    DATA_DIR=$2
    FRAME_DIR=$3
    VIDEO_DIR=$4
    COLOR_IMPORTANCE=$5
    NEARNESS_FACTOR=$6

    FOLDER_NAME="c${COLOR_IMPORTANCE}_n${NEARNESS_FACTOR}"

    #rm -r "$DATA_DIR/$FOLDER_NAME"

    if [ "$FLAGS" != "noencode" ]; then
        echo "Testing $DCT_START $DCT_END"
        echo "test" | tee "${LOG_DIR}/${FOLDER_NAME}.log"
        until try_and_fail ${SOURCE} "${DATA_DIR}/${FOLDER_NAME}" "${LOG_DIR}/${FOLDER_NAME}-numbers.csv" "${LOG_DIR}/${FOLDER_NAME}.log" $COLOR_IMPORTANCE $NEARNESS_FACTOR
        do
            echo "fuck"
        done
    fi
    echo "Done generating"
}

COLOR_IMP=$(echo $5 | cut -d ";" -f1 )
NEARNESS_FACT=$(echo $5 | cut -d ";" -f2 )

generate $1 $2 $3 $4 $COLOR_IMP $NEARNESS_FACT