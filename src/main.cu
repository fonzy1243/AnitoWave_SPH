#include <cstdio>
#include <bgfx/bgfx.h>
#include <bgfx/platform.h>
#include <GLFW/glfw3.h>

#ifdef _WIN64
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif

#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 900

static bool s_showStats = false;

static void glfw_errorCallback(int error, const char* description) {
    fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

static void glfw_keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_F1 && action == GLFW_RELEASE) {
        s_showStats = true;
    }
}

int main() {
    // GLFW window without OpenGL context
    glfwSetErrorCallback(glfw_errorCallback);
    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "AnitoWave_SPH", nullptr, nullptr);
    if (!window) return 1;
    glfwSetKeyCallback(window, glfw_keyCallback);
    // Calling bgfx::renderFrame to signal bgfx not to create a render thread
    bgfx::renderFrame();
    // Initialize bgfx
    bgfx::Init init;
    init.platformData.nwh = glfwGetWin32Window(window);

    int width, height;
    glfwGetWindowSize(window, &width, &height);
    init.type = bgfx::RendererType::Vulkan;
    init.resolution.width = width;
    init.resolution.height = height;
    init.resolution.reset = BGFX_RESET_VSYNC;
    if (!bgfx::init(init))
        return 1;
    // Set view 0 to window dimension
    const bgfx::ViewId kClearView = 0;
    bgfx::setViewClear(kClearView, BGFX_CLEAR_COLOR);
    bgfx::setViewRect(kClearView, 0, 0, bgfx::BackbufferRatio::Equal);
    // rendering
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        // Handle resize
        int oldWidth = width, oldHeight = height;
        glfwGetWindowSize(window, &oldWidth, &oldHeight);
        if (width != oldWidth || height != oldHeight) {
            bgfx::reset((uint32_t)width, (uint32_t)height, BGFX_RESET_VSYNC);
            bgfx::setViewRect(kClearView, 0, 0, bgfx::BackbufferRatio::Equal);
        }
        // Dummy draw call to make sure view 0 is cleared
        bgfx::touch(kClearView);
        // Advance to next frame.
        bgfx::frame();
    }
    bgfx::shutdown();
    glfwTerminate();
    return 0;
}