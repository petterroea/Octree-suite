#include "gui.h"

#include <iostream>
#include <exception>

#include "imgui.h"
#include "imgui_impl_sdl.h"
#include "imgui_impl_opengl3.h"

void GLAPIENTRY
MessageCallback( GLenum source,
                 GLenum type,
                 GLuint id,
                 GLenum severity,
                 GLsizei length,
                 const GLchar* message,
                 const void* userParam )
{
  fprintf( stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
           ( type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : "" ),
            type, severity, message );
}


void Gui::init() {
    //SDL
    int WIDTH = 800;
    int HEIGHT = 600;

    // SDL setup
    if(SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cout << "Failed initialize video" << std::endl;
        throw std::runtime_error("Failed to initialize video");
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);

    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    this->window = SDL_CreateWindow("Octree capture", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WIDTH, HEIGHT, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
    if (!this->window) {/* Die if creation failed */
        std::cout << "Unable to create window" << std::endl;
        throw std::runtime_error("Unable to create window");
    }

    SDL_SetWindowResizable(this->window, SDL_TRUE);

    SDL_GLContext gl_context = SDL_GL_CreateContext(this->window);

    SDL_GL_SetSwapInterval(1);

    // imgui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsClassic();

    // Setup Platform/Renderer backends
    ImGui_ImplSDL2_InitForOpenGL(this->window, gl_context);
    ImGui_ImplOpenGL3_Init("#version 130");

    // glew
    glewExperimental=true; // Needed in core profile
    if (glewInit() != GLEW_OK) {
        std::cout << "Failed to initialize GLEW" << std::endl;
        throw std::runtime_error("Failed to initialize GLEW");
    }

    // OpenGL init
    //glEnable              ( GL_DEBUG_OUTPUT );
    //glDebugMessageCallback( MessageCallback, 0 );

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.8, 0.8, 1.0, 1.0);

    std::cout << "Initialized graphics" << std::endl;
}

Gui::Gui() {
    this->init();
}

Gui::~Gui() {

}