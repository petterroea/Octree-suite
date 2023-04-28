#pragma once

class VideoEncoderRunArgs {
    int frameLimit = -1;
    float tree_nearness_factor = 0.9999f;
    float color_importance_factor = 0.0001f;

    int encodingThreadCount;
    int chunkConcurrencyCount = 1;

    int encodingChunkSize = 10;
public:
    VideoEncoderRunArgs();

    void printSettings() const;

    int getFrameLimit() const;
    void setFrameLimit(int frameLimit);

    float getTreeNearnessFactor() const;
    void setTreeNearnessFactor(float tree_nearness_factor);
 
    float getColorImportanceFactor() const;
    void setColorImportanceFactor(float color_importance_factor);

    int getEncodingThreadCount() const;
    void setEncodingThreadCount(int encodingThreadCount);

    int getChunkConcurrencyCount() const;
    void setChunkConcurrencyCount(int chunkConcurrencyCount);

    int getEncodingChunkSize() const;
    void setEncodingChunkSize(int chunksize);
};