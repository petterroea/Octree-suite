SOURCE="datasets/wave/oct"

FLAGS="$6"

function try_and_fail() {
    SOURCE=$1
    DIR=$2
    STATS_FILE=$3
    LOG_DIR=$4
    CHUNK_SIZE=$5

    start=`date +%s`
    bin/octreeVideoEncoder $SOURCE $DIR --chunk_size $CHUNK_SIZE --compression_stats $STATS_FILE --encoding_threads 1 --limit $CHUNK_SIZE --quantization_start 1 --quantization_end 1 | tee $LOG_DIR
    end=`date +%s`
    echo Execution time was `expr $end - $start` seconds. > "$LOG_DIR.time"

}

function generate() {
    LOG_DIR=$1
    DATA_DIR=$2
    FRAME_DIR=$3
    VIDEO_DIR=$4
    CHUNK_SIZE=$5

    FOLDER_NAME="chunk_${CHUNK_SIZE}"

    if [ "$FLAGS" != "noencode" ]; then
        echo "Testing $DCT_START $DCT_END"
        echo "test" | tee "${LOG_DIR}/${FOLDER_NAME}.log"
        until try_and_fail ${SOURCE} "${DATA_DIR}/${FOLDER_NAME}" "${LOG_DIR}/${FOLDER_NAME}-numbers.csv" "${LOG_DIR}/${FOLDER_NAME}.log" $CHUNK_SIZE
        do
            echo "fuck"
        done
    fi
    echo "Done generating"
}

generate $1 $2 $3 $4 $5