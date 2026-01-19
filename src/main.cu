#include <cstdio>
#include <bx/bx.h>
#include <bgfx/bgfx.h>
#include <bgfx/platform.h>
#include <GLFW/glfw3.h>

#ifdef _WIN64
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif

class AnitoWave {
public:
    struct Config {
        const char* title = "AnitoWave";
        uint32_t width = 1600;
        uint32_t height = 900;
        bgfx::RendererType::Enum rendererType = bgfx::RendererType::Vulkan;
        bool vsync = true;
    };

    AnitoWave(const Config& config = Config());
    virtual ~AnitoWave();

    // Initialize GLFW and bgfx
    bool init();
    // Main application loop
    void run();

    uint32_t getWidth() const { return m_width; }
    uint32_t getHeight() const { return m_height; }
    GLFWwindow* getWindow() const { return m_window; }
    bgfx::ViewId getClearView() const { return m_kClearView; }
private:
    static void glfwErrorCallback(int error, const char* description);
    static void glfwKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

    Config m_config;
    GLFWwindow* m_window = nullptr;
    uint32_t m_width;
    uint32_t m_height;
    bool m_showStats = false;
    const bgfx::ViewId m_kClearView = 0;
};

AnitoWave::AnitoWave(const Config &config) : m_config(config), m_width(config.width), m_height(config.height) {
}

AnitoWave::~AnitoWave() {
    if (m_window) {
        bgfx::shutdown();
        glfwTerminate();
    }
}

bool AnitoWave::init() {
    // GLFW window without OpenGL context
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return false;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    m_window = glfwCreateWindow(m_width, m_height, m_config.title, nullptr, nullptr);
    if (!m_window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return false;
    }
    glfwSetWindowUserPointer(m_window, this);
    glfwSetKeyCallback(m_window, glfwKeyCallback);
    // Calling bgfx::renderFrame to signal bgfx not to create a render thread
    bgfx::renderFrame();
    // Initialize bgfx
    bgfx::Init init;
#ifdef _WIN64
    init.platformData.nwh = glfwGetWin32Window(m_window);
#endif

    glfwGetWindowSize(m_window, (int*)&m_width, (int*)&m_height);
    init.type = m_config.rendererType;
    init.resolution.width = m_config.width;
    init.resolution.height = m_config.height;
    init.resolution.reset = m_config.vsync ? BGFX_RESET_VSYNC : BGFX_RESET_NONE;
    if (!bgfx::init(init)) {
        fprintf(stderr, "Failed to initialize bgfx\n");
        glfwDestroyWindow(m_window);
        glfwTerminate();
        return false;
    }
    // Set view 0 to window dimension
    bgfx::setViewClear(m_kClearView, BGFX_CLEAR_COLOR);
    bgfx::setViewRect(m_kClearView, 0, 0, bgfx::BackbufferRatio::Equal);

    return true;
}

void AnitoWave::run() {
    while (!glfwWindowShouldClose(m_window)) {
        glfwPollEvents();
        // Handle resize
        int newWidth, newHeight;
        glfwGetWindowSize(m_window, &newWidth, &newHeight);
        if (m_width != newWidth || m_height != newHeight) {
            m_width = newWidth;
            m_height = newHeight;
            bgfx::reset(m_width, m_height, m_config.vsync ? BGFX_RESET_VSYNC : BGFX_RESET_NONE);
            bgfx::setViewRect(m_kClearView, 0, 0, bgfx::BackbufferRatio::Equal);
        }
        // Simulation and rendering

        bgfx::touch(m_kClearView);
        bgfx::setDebug(m_showStats ? BGFX_DEBUG_STATS : BGFX_DEBUG_TEXT);
        bgfx::frame();
    }
}

void AnitoWave::glfwErrorCallback(int error, const char *description) {
    fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

void AnitoWave::glfwKeyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    AnitoWave* app = static_cast<AnitoWave *>(glfwGetWindowUserPointer(window));
    if (app && key == GLFW_KEY_F1 && action == GLFW_RELEASE) {
        app->m_showStats = !app->m_showStats;
    }
}

int main() {
    AnitoWave::Config config;
    config.title = "AnitoWave SPH";

    AnitoWave app(config);
    if (!app.init()) {
        return 1;
    }

    app.run();

    return 0;
}