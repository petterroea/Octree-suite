#pragma once

struct VideoMode {
    int colorWidth;
    int colorHeight;
    int depthWidth;
    int depthHeight;
};

enum RenderMode {
    OPENGL,
    HEADLESS
};
