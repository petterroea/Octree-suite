#include "videoEncoderRunArgs.h"

#include <iostream>
#include <thread>

VideoEncoderRunArgs::VideoEncoderRunArgs(): encodingThreadCount(std::thread::hardware_concurrency()) {

}

void VideoEncoderRunArgs::printSettings() const {
    std::cout << "Encoder settings" << std::endl;
    std::cout << " Frame limit: " << this->getFrameLimit() << std::endl;
    std::cout << " Chunk size: " << this->getEncodingChunkSize() << std::endl;
    std::cout << "" << std::endl;
    std::cout << " Encoding threads: " << this->getEncodingThreadCount() << std::endl;
    std::cout << " Chunk concurrency: " << this->getChunkConcurrencyCount() << std::endl;
    std::cout << "" << std::endl;
    std::cout << " Nearness factor: " << this->getTreeNearnessFactor() << std::endl;
    std::cout << " Color importance: " << this->getColorImportanceFactor() << std::endl;
    std::cout << "" << std::endl;
    std::cout << " Quantization start: " << this->getQuantizationStart() << std::endl;
    std::cout << " Quantization end: " << this->getQuantizationEnd() << std::endl;
    std::cout << "" << std::endl;
    std::cout << " Should skip reduction? " << (this->getShouldSkipReduction() ? "yes" : "no") << std::endl;
    std::cout << "" << std::endl;
    std::cout << " Stats output: " << this->getCompressionStatsOutput() << std::endl;
}

int VideoEncoderRunArgs::getFrameLimit() const { 
    return frameLimit; 
}

void VideoEncoderRunArgs::setFrameLimit(int frameLimit) { 
    this->frameLimit = frameLimit; 
}

float VideoEncoderRunArgs::getTreeNearnessFactor() const { 
    return tree_nearness_factor; 
}
void VideoEncoderRunArgs::setTreeNearnessFactor(float tree_nearness_factor) { 
    this->tree_nearness_factor = tree_nearness_factor; 
}

float VideoEncoderRunArgs::getColorImportanceFactor() const { 
    return color_importance_factor; 
}
void VideoEncoderRunArgs::setColorImportanceFactor(float color_importance_factor) { 
    this->color_importance_factor = color_importance_factor; 
}

int VideoEncoderRunArgs::getEncodingThreadCount() const {
    return this->encodingThreadCount;
}
void VideoEncoderRunArgs::setEncodingThreadCount(int encodingThreadCount) {
    this->encodingThreadCount = encodingThreadCount;
}

int VideoEncoderRunArgs::getChunkConcurrencyCount() const {
    return this->chunkConcurrencyCount;
}
void VideoEncoderRunArgs::setChunkConcurrencyCount(int chunkConcurrencyCount) {
    this->chunkConcurrencyCount = chunkConcurrencyCount;
}

int VideoEncoderRunArgs::getEncodingChunkSize() const {
    return this->encodingChunkSize;
}
void VideoEncoderRunArgs::setEncodingChunkSize(int chunksize) {
    this->encodingChunkSize = chunksize;
}

int VideoEncoderRunArgs::getQuantizationStart() const {
    return this->quantizationStart;
}

void VideoEncoderRunArgs::setQuantizationStart(int quantizationStart) {
    this->quantizationStart = quantizationStart;
}

int VideoEncoderRunArgs::getQuantizationEnd() const {
    return this->quantizationEnd;
}

void VideoEncoderRunArgs::setQuantizationEnd(int quantizationEnd) {
    this->quantizationEnd = quantizationEnd;
}

bool VideoEncoderRunArgs::getShouldSkipReduction() const {
    return this->shouldSkipReduction;
}
void VideoEncoderRunArgs::setShouldSkipReduction(bool shouldSkipReduction) {
    this->shouldSkipReduction = shouldSkipReduction;
}

char* VideoEncoderRunArgs::getCompressionStatsOutput() const {
    return this->compressionStatsOutput;
}

void VideoEncoderRunArgs::setCompressionStatsOutput(char* compressionStatsOutput) {
    this->compressionStatsOutput = compressionStatsOutput;
}