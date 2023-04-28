#include "encodingJob.h"

EncodingJob::EncodingJob(int from, int to, std::filesystem::path fullSequencePath): from(from), to(to), fullSequencePath(fullSequencePath) {

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