#pragma once

#include "pointcloudRenderer.h"
#include "depthCamera.h"

struct PointcloudCameraRendererPair {
    PointcloudRenderer* renderer;
    DepthCamera* camera;
    bool capture;
};