#include <iostream>
#include <chrono>

#include <SDL.h>
#include <GL/glew.h>

#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>

#include <librealsense2/rs.hpp>
#include <librealsense2/hpp/rs_internal.hpp>

#include "imgui.h"
#include "imgui_impl_sdl.h"
#include "imgui_impl_opengl3.h"

#include "main.h"

#include "render/texturedPointcloudRenderer.h"
#include "depthCamera/depthCamera.h"
#include "depthCamera/drivers/realsenseDepthCamera.h"

#include "capture/capturer.h"

#include "depthCamera/openCVCalibrator.h"

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


int main(int argc, char** argv) {
    int WIDTH = 800;
    int HEIGHT = 600;

    // SDL setup
    if(SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cout << "Failed initialize video" << std::endl;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);

    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_Window* mainwindow = SDL_CreateWindow("Octree capture", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WIDTH, HEIGHT, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
    if (!mainwindow) {/* Die if creation failed */
        std::cout << "Unable to create window" << std::endl;
        return -1;
    }

    SDL_SetWindowResizable(mainwindow, SDL_TRUE);

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

    // Setup glew
    glewExperimental=true; // Needed in core profile
    if (glewInit() != GLEW_OK) {
        std::cout << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    glEnable              ( GL_DEBUG_OUTPUT );
    glDebugMessageCallback( MessageCallback, 0 );

    glEnable(GL_DEPTH_TEST);

    // Setup librealsense
    rs2::context ctx;

    std::cout << "hello from librealsense - " << RS2_API_VERSION_STR << std::endl;
    std::cout << "You have " << ctx.query_devices().size() << " RealSense devices connected" << std::endl;

    PointcloudShader* shader = new PointcloudShader();

    std::vector<DepthCamera*> deviceList;
    bool first = true;
    for(auto&& dev : ctx.query_devices()) {
        std::cout << "Creating device" << std::endl;
        RealsenseDepthCamera* camera = new RealsenseDepthCamera(RenderMode::OPENGL, dev, first);
        first = false;

        camera->startCaptureThread();
        deviceList.push_back(camera);
    }

    Capturer capturer(deviceList);

    // Render loop
    glClearColor(0.8, 0.8, 1.0, 1.0);
    bool should_run = true;

    // For measuring frame time
    static float values[90] = {};
    static int values_offset = 0;

    float yaw = 0.0f;
    float pitch = 0.0f;

    float storedYaw = 0.0f;
    float storedPitch = 0.0f;

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

        SDL_GetWindowSize(mainwindow, &WIDTH, &HEIGHT);
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

        capturer.displayGui();
        capturer.getFrame();
        capturer.render(view, projection);
/*
        for(auto device : deviceList) {
            DepthCamera* camera = device->camera;
            // Waits for the frame processing to be complete. Actions performed after this should be thread safe
            camera->waitForNewFrame();

            ImGui::PushID(camera->getSerial().c_str());
            ImGui::Text(("RealSense device: " + camera->getSerial()).c_str());
            PointcloudRenderer* renderer = device->renderer;

            if(device->capture) {
                camera->uploadTextures();
                rs2::points pointcloud = camera->getLastPointcloud();
                renderer->updateData(pointcloud);
            }
            //rs2::frameset frame = camera->processFrame();
            //camera->drawImmediateGui();

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, camera->getColorTextureHandle());
            renderer->render(camera->getCalibration(), view, projection);

            ImGui::Checkbox("Capture", &device->capture);
            camera->drawImmediateGui();
            if(ImGui::CollapsingHeader("Textures")) {
                ImGui::Image((void*)(intptr_t)camera->getDepthTextureHandle(), ImVec2(600, 400));
                ImGui::SameLine();
                ImGui::Image((void*)(intptr_t)camera->getColorTextureHandle(), ImVec2(600, 400));
            }
            ImGui::Separator();
            ImGui::PopID();
        }
*/

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

        //capture.displayGui(deviceList);
        //capture.renderHelpLines(view, projection);


        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        auto end = std::chrono::system_clock::now();
        float elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        values[values_offset] = elapsed_time;
        values_offset = (values_offset + 1) % IM_ARRAYSIZE(values);
        SDL_GL_SwapWindow(mainwindow);
    }

    for(auto device : deviceList) {
        device->endCaptureThread();
    }
    for(auto device : deviceList) {
        device->waitForThreadJoin();
        delete device;
    }

    delete shader;

    return 0;
}