#pragma once

#include <SDL.h>
#include <GL/glew.h>

class Gui {
    SDL_Window* window;
    void init();
public:
    Gui();
    ~Gui();

    SDL_Window* getWindow() { return window; }
};