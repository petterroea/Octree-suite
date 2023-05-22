#pragma once

#include <filesystem>

class EncodingJob {
    int from;
    int to;

    std::filesystem::path fullSequencePath;
public:
    EncodingJob(int from, int to, std::filesystem::path fullSequencePath);
    ~EncodingJob();

    int getFrom() const;
    int getTo() const;
    std::filesystem::path getFullSequencePath() const;
};
