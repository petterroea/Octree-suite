#include "octreeSequence.h"
#include "encoding/octreeSequenceEncoder.h"

#include "videoEncoderRunArgs.h"

#include <dct2d/dct.h>
#include <dct2d/quantization.h>

#include <iostream>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

void printUsage() {
    std::cout << "octreeVideoEncoder - encodes octree video" << std::endl;
    std::cout << "Usage:" << std::endl;
    std::cout << "    octreeVideoEncoder [octreeFolder] [outputFolder] [concurrency_sequences] [encoding_threads]" << std::endl;
    std::cout << "Optional arguments:" << std::endl;
    std::cout << "  --limit [integer] Specifies the amount of frames to encode" << std::endl;
    std::cout << "  --chunk_size [integer] The amount of frames that are bundled together when trying to eliminate duplicates" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "  --encoding_threads [integer] The number of threads to use to search for duplicate nodes within a chunk" << std::endl;
    std::cout << "  --chunk_concurrency [integer] The number of chunks to process at the same time. Slightly overprovisioning the number of threads you have available may improve performance" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "  --nearness_factor [float] Specifies how \"close\" two trees have to be to be considered mergable" << std::endl;
    std::cout << "  --color_importance [float] Specifies how much color difference(rgb euclidean) affects the nearness factor in the formula nearness / ( color_dist * color_importance)" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "  --quantization_start [int] Specifies the start of a generated quantization gradient" << std::endl;
    std::cout << "  --quantization_end [int] Specifies the start of a generated quantization gradient" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "  --skip_reduction [true/false] Specifies whether or not to skip reduction" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "  --compression_stats [path] Specifies a path to be used to output compression stats" << std::endl;
}

int main(int argc, char** argv) {
    if(argc < 3 || argc % 2 != 1) {
        printUsage();
        return 1;
    }

    // Parse optional args
    VideoEncoderRunArgs arguments;
    for(int i = 3; i < argc-1; i+= 2) {
        if(!strcmp("--limit", argv[i])) {
            arguments.setFrameLimit(atoi(argv[i+1]));
        } else if(!strcmp("--chunk_size", argv[i])) {
            arguments.setEncodingChunkSize(atof(argv[i+1]));
        } else if(!strcmp("--encoding_threads", argv[i])) {
            arguments.setEncodingThreadCount(atof(argv[i+1]));
        } else if(!strcmp("--chunk_concurrency", argv[i])) {
            arguments.setChunkConcurrencyCount(atof(argv[i+1]));
        } else if(!strcmp("--nearness_factor", argv[i])) {
            arguments.setTreeNearnessFactor(atof(argv[i+1]));
        } else if(!strcmp("--color_importance", argv[i])) {
            arguments.setColorImportanceFactor(atof(argv[i+1]));
        } else if(!strcmp("--quantization_start", argv[i])) {
            arguments.setQuantizationStart(atoi(argv[i+1]));
        } else if(!strcmp("--quantization_end", argv[i])) {
            arguments.setQuantizationEnd(atoi(argv[i+1]));
        } else if(!strcmp("--skip_reduction", argv[i])) {
            arguments.setShouldSkipReduction(!strcmp("true", argv[i+1]));
        } else if(!strcmp("--compression_stats", argv[i])) {
            arguments.setCompressionStatsOutput(argv[i+1]);
        } else {
            std::cout << "Unknown option: " << argv[i] << std::endl;
            return 1;
        }

    }

    arguments.printSettings();

    build_lookup_table();

    OctreeSequence* sequence = new OctreeSequence(argv[1]);

    std::cout << "Loaded octree sequence with " << sequence->getFrameCount() << " frames @ " << sequence->getFps() << " fps." << std::endl;

    std::filesystem::path outputFolder(argv[2]);

    int concurrency = atoi(argv[3]);
    int encodingThreads = atoi(argv[4]);

    OctreeSequenceEncoder* encoder = new OctreeSequenceEncoder(sequence, outputFolder, &arguments);
    encoder->encode();

    delete sequence;
    delete encoder;

    std::cout << "Done encoding, exiting..." << std::endl;
    return 0;
}