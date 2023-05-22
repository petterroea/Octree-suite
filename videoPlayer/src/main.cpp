#include <iostream>
#include <filesystem>
#include <chrono>

#include <cstring>

#include "players/videoPlayer.h"
#include "players/octree/octreeVideoPlayer.h"
#include "players/ply/plyVideoPlayer.h"

#include "time/timeProvider.h"
#include "time/clockTimeProvider.h"
#include "time/recordingTimeProvider.h"

#include "gui/gui.h"

#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>

#include "imgui.h"
#include "imgui_impl_sdl.h"
#include "imgui_impl_opengl3.h"

// Linux only!
#include <unistd.h>

void printUsage() {
    std::cout << "Usage: " << std::endl;
    std::cout << "  videoPlayer [format] [file] <options>" << std::endl;
    std::cout << "Supported formats: " << std::endl;
    std::cout << "  octree - Octree video format" << std::endl;
    std::cout << "  ply - PLY sequence(with metadata)" << std::endl;
    std::cout << std::endl;
    std::cout << "Supported options: " << std::endl;
    std::cout << "  --record [outdir] - records the entire video to PNG frames" << std::endl;
    std::cout << "  --width [number] - specifies the window width" << std::endl;
    std::cout << "  --height [number] - specifies the window height" << std::endl;
}

int main(int argc, char *argv[]) {
    std::cout << "Octree video renderer" << std::endl;

    if(argc < 3 || argc % 2 != 1) {
        printUsage();
        return 1;
    }

    auto format = argv[1];
    auto file = argv[2];

    TimeProvider* timeProvider = new ClockTimeProvider();
    bool recordMode = false;
    std::filesystem::path* recordingOutDir = nullptr;

    int WIDTH = 800;
    int HEIGHT = 600;

    for(int i = 3; i < argc-1; i += 2) {
        if(!strcmp(argv[i], "--record")) {
            std::cout << "Starting in recording mode, recording to " << argv[i+1] << std::endl;
            recordMode = true;

            delete timeProvider;
            timeProvider = new RecordingTimeProvider(30.0f, 11);
            //timeProvider = new RecordingTimeProvider(30.0f, 30);

            recordingOutDir = new std::filesystem::path(argv[i+1]);
            std::filesystem::create_directories(*recordingOutDir);
        } else if(!strcmp(argv[i], "--width")) {
            WIDTH = atoi(argv[i+1]);
        } else if(!strcmp(argv[i], "--height")) {
            HEIGHT = atoi(argv[i+1]);
        } else {
            throw std::runtime_error("Unknown parameter " + std::string(argv[i]));
        }
    }

    VideoPlayer* player = nullptr;

    // Init GUI
    Gui* gui = new Gui(WIDTH, HEIGHT, "Octree playback application");

    if(!strcmp(format, "octree")) {
        player = new OctreeVideoPlayer(timeProvider, std::filesystem::path(file));
    } else if(!strcmp(format, "ply")) {
        player = new PlyVideoPlayer(timeProvider, std::filesystem::path(file));
    } else {
        throw std::runtime_error("Invalid format: " + std::string(format));
    }

    // Render loop
    bool should_run = true;

    // For measuring frame time
    static float values[90] = {};
    static int values_offset = 0;

    float yaw = 1.0f;
    float pitch = 0.0f;

    float storedYaw = 30.0f;
    float storedPitch = 30.0f;

    float zoom = 1.0f;

    bool tracking_mouse = 0;
    glm::ivec2 mouse_pos;

    ImGuiIO& imguiIo = ImGui::GetIO();

    while(should_run) {
        auto start = std::chrono::system_clock::now();
        // First listen for events from SDL
        SDL_Event event;
        while(SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if(imguiIo.WantCaptureMouse || imguiIo.WantCaptureKeyboard) {
                continue;
            }
            int x, y;
            switch(event.type) {
                case SDL_KEYUP:
                    if(event.key.keysym.sym == SDLK_ESCAPE) {
                        should_run = false;
                    }
                    break;
                case SDL_QUIT:
                    should_run = false;
                    break;
                case SDL_MOUSEBUTTONDOWN:
                    tracking_mouse = true;
                    SDL_GetMouseState(&x, &y);
                    mouse_pos.x = x;
                    mouse_pos.y = y;
                    break;
                case SDL_MOUSEMOTION:
                    if(tracking_mouse) {
                        SDL_GetMouseState(&x, &y);
                        yaw = static_cast<float>(x-mouse_pos.x)*0.1f;
                        pitch = static_cast<float>(y-mouse_pos.y)*0.1f;
                    }
                    break;
                case SDL_MOUSEBUTTONUP:
                    tracking_mouse = false;
                    storedYaw += yaw;
                    storedPitch += pitch;
                    yaw = 0.0f;
                    pitch = 0.0f;
                    break;
                case SDL_MOUSEWHEEL:
                    if(event.wheel.y > 0) {
                        zoom = zoom * 1.1f;
                    } else {
                        zoom = zoom / 1.1f;
                    }
                    break;
                default:
                    //do nothing
                    {}
            }
        }

        SDL_GetWindowSize(gui->getWindow(), &WIDTH, &HEIGHT);
        glViewport(0, 0, WIDTH, HEIGHT);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        glm::mat4 model(1.0f);
        glm::mat4 view = glm::lookAt(
            // Temp bs
            glm::rotateY(glm::rotateX(glm::vec3(0.0f, 0.0f, 1.0f), glm::radians(pitch+storedPitch)), glm::radians(yaw+storedYaw))*zoom,
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, -1.0f, 0.0f)
        );
        glm::mat4 projection = glm::perspective(
            glm::radians(75.0f),
            (float)WIDTH/(float)HEIGHT,
            0.1f, 
            100.0f
        );

        std::cout << "======== FRAME" << std::endl;

        // Renders a frame
        player->render(WIDTH, HEIGHT, view, projection);

        if(!recordMode) {
            ImGui::Begin("Performance");

            float average = 0.0f;
            float min = 1000.0f;
            float max = 0.0f;
            for (int n = 0; n < IM_ARRAYSIZE(values); n++) {
                average += values[n];
                if(values[n] > max)
                    max = values[n];
                if(values[n] < min) 
                    min = values[n];
            }

            average /= (float)IM_ARRAYSIZE(values);
            char overlay[128];
            sprintf(overlay, "min %fms avg %fms (%f fps) max %fms", min, average, 1000.0f/average, max);
            ImGui::PlotLines("Frame time", values, IM_ARRAYSIZE(values), values_offset, overlay, 0.0f, max*1.1f, ImVec2(0, 80.0f));

            ImGui::End();

            timeProvider->renderControls();
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if(!player->isBuffering()) {
            if(recordMode) {
                gui->saveFramebufferToFile(
                    reinterpret_cast<RecordingTimeProvider*>(timeProvider)->getCurrentFrame(),
                    recordingOutDir
                );
            }
            timeProvider->onFrameComplete();
        }

        auto end = std::chrono::system_clock::now();
        float elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        values[values_offset] = elapsed_time;
        values_offset = (values_offset + 1) % IM_ARRAYSIZE(values);
        SDL_GL_SwapWindow(gui->getWindow());
    }

    // Shut down
    std::cout << "Shutting down..." << std::endl;
    delete gui;
    delete player;
    delete timeProvider;
}