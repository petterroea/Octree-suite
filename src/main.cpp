#include <iostream>
#include <chrono>

#include <SDL.h>
#include <GL/glew.h>

#include "imgui.h"
#include "imgui_impl_sdl.h"
#include "imgui_impl_opengl3.h"


int main(int argc, char** argv) {
    int WIDTH = 800;
    int HEIGHT = 600;

    if(SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cout << "Failed initialize video" << std::endl;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);

    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_Window* mainwindow = SDL_CreateWindow("Realsense demo", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WIDTH, HEIGHT, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
    if (!mainwindow) {/* Die if creation failed */
        std::cout << "Unable to create window" << std::endl;
        return -1;
    }

    SDL_GLContext gl_context = SDL_GL_CreateContext(mainwindow);

    SDL_GL_SetSwapInterval(1);

    //imgui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsClassic();

    // Setup Platform/Renderer backends
    ImGui_ImplSDL2_InitForOpenGL(mainwindow, gl_context);
    ImGui_ImplOpenGL3_Init("#version 130");

    glewExperimental=true; // Needed in core profile
    if (glewInit() != GLEW_OK) {
        std::cout << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    glClearColor(0.8, 0.8, 1.0, 1.0);
    bool should_run = true;

    auto start = std::chrono::system_clock::now();
    while(should_run) {
        // First listen for events from SDL
        SDL_Event event;
        while(SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            switch(event.type) {
                case SDL_KEYUP:
                    if(event.key.keysym.sym == SDLK_ESCAPE) {
                        should_run = false;
                    }
                case SDL_QUIT:
                    should_run = false;
                default:
                    //do nothing
                    {}
            }
        }
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        // GUI
        ImGui::Begin("RealSense test");

        ImGui::Text("Hello");
        ImGui::End();


        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        SDL_GL_SwapWindow(mainwindow);
    }


    return 0;
}