#pragma once

#include <filesystem>

#include <SDL.h>
#include <GL/glew.h>

class Gui {
    SDL_Window* window;
    void init(int width, int height, const char* windowTitle);
public:
    Gui(int width, int height, const char* windowTitle);
    ~Gui();

    SDL_Window* getWindow() { return window; }
    void saveFramebufferToFile(int frameNumber, std::filesystem::path* folder);
};