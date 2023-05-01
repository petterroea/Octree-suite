#include "gui.h"

#include <iostream>
#include <exception>
#include <vector>

#include "imgui.h"
#include "imgui_impl_sdl.h"
#include "imgui_impl_opengl3.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

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


void Gui::init(int WIDTH, int HEIGHT, const char* windowTitle) {
    // SDL setup
    if(SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cout << "Failed initialize video" << std::endl;
        throw std::runtime_error("Failed to initialize video");
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);

    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    this->window = SDL_CreateWindow(windowTitle, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
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

Gui::Gui(int width, int height, const char* windowTitle) {
    this->init(width, height, windowTitle);
}

Gui::~Gui() {

}
void Gui::saveFramebufferToFile(int frameNumber, std::filesystem::path* folder) {
    // https://lencerf.github.io/post/2019-09-21-save-the-opengl-rendering-to-image-file/
    int width, height;

    std::filesystem::path filename = *folder / std::filesystem::path("output-" + std::to_string(frameNumber) + ".png");

    SDL_GetWindowSize(this->window, &width, &height);

    GLsizei nrChannels = 3;
    GLsizei stride = nrChannels * width;
    stride += (stride % 4) ? (4 - stride % 4) : 0;
    GLsizei bufferSize = stride * height;
    std::vector<char> buffer(bufferSize);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glReadBuffer(GL_FRONT);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, buffer.data());
    stbi_flip_vertically_on_write(true);
    stbi_write_png(filename.string().c_str(), width, height, nrChannels, buffer.data(), stride);

    std::cout << "Saved frame " << frameNumber << " to " << filename.string() << std::endl;
}