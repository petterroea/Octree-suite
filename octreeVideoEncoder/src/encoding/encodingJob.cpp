#include "encodingJob.h"

#include <iostream>

EncodingJob::EncodingJob(int from, int to, std::filesystem::path fullSequencePath): from(from), to(to), fullSequencePath(fullSequencePath) {

}

EncodingJob::~EncodingJob() {
    std::cout << "EncodingJob destruct: " << this->from << " " << this->to << std::endl;
}

int EncodingJob::getFrom() const {
    return this->from;
}

int EncodingJob::getTo() const {
    return this->to;
}

std::filesystem::path EncodingJob::getFullSequencePath() const {
    return this->fullSequencePath;
}