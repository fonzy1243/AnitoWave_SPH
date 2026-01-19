#include <cstdio>
#include <vector>
#include <bx/bx.h>
#include <bx/math.h>
#include <bgfx/bgfx.h>
#include <bgfx/platform.h>
#include <GLFW/glfw3.h>

#ifdef _WIN64
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif

struct PosColorVertex {
    float x, y, z;
    uint32_t abgr;
};

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

    void initParticleRendering();
    void renderParticles();
    void generateCircleTemplate(float radius, int segments);
    void setupOrtho();

    Config m_config;
    GLFWwindow* m_window = nullptr;
    uint32_t m_width;
    uint32_t m_height;
    bool m_showStats = false;
    const bgfx::ViewId m_kClearView = 0;

    // Particle rendering data
    bgfx::VertexLayout m_particleLayout;
    std::vector<PosColorVertex> m_circleTemplate;
    std::vector<uint16_t> m_circleIndices;

    // SPH particle data
    std::vector<float> m_particlePositions;
    std::vector<uint32_t> m_particleColors;
    float m_particleRadius = 0.02f;
};

AnitoWave::AnitoWave(const Config &config) : m_config(config), m_width(config.width), m_height(config.height) {
}

AnitoWave::~AnitoWave() {
    if (m_window) {
        bgfx::shutdown();
        glfwTerminate();
    }
}

void AnitoWave::initParticleRendering() {
    m_particleLayout.begin()
        .add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
        .add(bgfx::Attrib::Color0, 4, bgfx::AttribType::Uint8)
        .end();

    generateCircleTemplate(1.0f, 32);

    const int numParticles = 100;
    m_particlePositions.reserve(numParticles * 2);
    m_particleColors.reserve(numParticles);

    for (int i = 0; i < numParticles; ++i) {
        float angle = (float)i / numParticles * bx::kPi * 2.0f;
        float radius = 0.3f + 0.2f * bx::sin(angle * 3.0f);
        m_particlePositions.push_back(bx::cos(angle) * radius);
        m_particlePositions.push_back(bx::sin(angle) * radius);

        uint8_t r = 0;
        uint8_t g = 0;
        uint8_t b = 255;
        m_particleColors.push_back(0xff000000 | (b << 16) | (g << 8) | r);
    }
}

void AnitoWave::generateCircleTemplate(float radius, int segments) {
    m_circleTemplate.clear();
    m_circleIndices.clear();

    m_circleTemplate.push_back({0.f, 0.f, 0.f, 0xffffffff});

    for (int i = 0; i <= segments; ++i) {
        float angle = (float)i / segments * bx::kPi * 2.0f;
        float x = bx::cos(angle) * radius;
        float y = bx::sin(angle) * radius;
        m_circleTemplate.push_back({x, y, 0.f, 0xffffffff});
    }

    for (int i = 0; i < segments; ++i) {
        m_circleIndices.push_back(0);
        m_circleIndices.push_back(i + 1);
        m_circleIndices.push_back(i + 2);
    }
}

void AnitoWave::setupOrtho() {
    const float aspect = (float)m_width / (float)m_height;
    float proj[16];
    bx::mtxOrtho(proj, -aspect, aspect, -1.0f, 1.0f, 0.0f, 100.0f, 0.0f, bgfx::getCaps()->homogeneousDepth);
    bgfx::setViewTransform(m_kClearView, nullptr, proj);
}

void AnitoWave::renderParticles() {
    std::vector<PosColorVertex> vertices;
    std::vector<uint16_t> indices;

    const size_t vertsPerCircle = m_circleTemplate.size();
    const size_t indicesPerCircle = m_circleIndices.size();
    const size_t numParticles = m_particlePositions.size() / 2;

    vertices.reserve(vertsPerCircle * numParticles);
    indices.reserve(indicesPerCircle * numParticles);

    for (size_t i = 0; i < numParticles; ++i) {
        float px = m_particlePositions[i * 2];
        float py = m_particlePositions[i * 2 + 1];
        uint32_t color = m_particleColors[i];

        uint16_t baseVertex = vertices.size();

        for (const auto& v : m_circleTemplate) {
            vertices.push_back({
                px + v.x * m_particleRadius,
                py + v.y * m_particleRadius,
                v.z,
                color
            });
        }

        for (uint16_t idx: m_circleIndices) {
            indices.push_back(baseVertex + idx);
        }
    }

    if (vertices.empty()) return;

    bgfx::TransientVertexBuffer tvb{};
    bgfx::TransientIndexBuffer tib{};

    if (bgfx::allocTransientBuffers(&tvb, m_particleLayout, vertices.size(), &tib, indices.size())) {
        bx::memCopy(tvb.data, vertices.data(), vertices.size() * sizeof(PosColorVertex));
        bx::memCopy(tib.data, indices.data(), indices.size() * sizeof(uint16_t));

        bgfx::setVertexBuffer(0, &tvb);
        bgfx::setIndexBuffer(&tib);
        bgfx::setState(BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A | BGFX_STATE_BLEND_ALPHA);
        bgfx::submit(m_kClearView, BGFX_INVALID_HANDLE);
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

    initParticleRendering();
    setupOrtho();

    return true;
}

void AnitoWave::run() {
    float time = 0.0f;

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

        bgfx::touch(m_kClearView);
        // Simulation and rendering
        time += 0.016f;
        const size_t numParticles = m_particlePositions.size() / 2;
        for (size_t i = 0; i < numParticles; ++i) {
            float angle = (float)i / numParticles * bx::kPi * 2.0f + time;
            float radius = 0.3f + 0.2f * bx::sin(angle * 3.0f + time * 2.0f);
            m_particlePositions[i * 2] = bx::cos(angle) * radius;
            m_particlePositions[i * 2 + 1] = bx::sin(angle) * radius;
        }

        renderParticles();

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