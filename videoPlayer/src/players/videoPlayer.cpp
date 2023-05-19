#include "videoPlayer.h"

VideoPlayer::VideoPlayer(TimeProvider* timeProvider) : timeProvider(timeProvider) {

}

int VideoPlayer::getCurrentFrame() {
    VideoMetadata metadata;
    this->getVideoMetadata(&metadata);

    float time = this->timeProvider->getTime();
    return static_cast<int>(time * metadata.fps);
}