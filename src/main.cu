#include <cstdio>
#include <vector>
#include <imgui/imgui.h>
#include <bx/bx.h>
#include <bx/math.h>
#include <bgfx/bgfx.h>
#include <bgfx/platform.h>
#include <GLFW/glfw3.h>

#include "bgfx_utils.h"
#include "../include/sph.cuh"

#ifdef _WIN64
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif

struct Particle {
    float2 pos;
    float2 vel;
};

struct PosColorVertex {
    float x, y, z;
    uint32_t abgr;
};

class AnitoWave {
public:
    struct Config {
        const char* title = "AnitoWave";
        uint32_t width = 1920;
        uint32_t height = 1080;
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
    static void glfwScrollCallback(GLFWwindow* window, double xoffset, double yoffset);

    void initParticleRendering();
    void renderParticles();
    void renderImGui();
    void generateCircleTemplate(float radius, int segments);
    void setupOrtho();

    Config m_config;
    GLFWwindow* m_window = nullptr;
    uint32_t m_width;
    uint32_t m_height;
    int32_t m_scroll = 0;
    bool m_showStats = false;
    bool m_showParamEditor = true;
    const bgfx::ViewId m_kClearView = 0;

    // bgfx programs
    bgfx::ProgramHandle m_program;

    // Particle rendering data
    bgfx::VertexLayout m_particleLayout;
    std::vector<PosColorVertex> m_circleTemplate;
    std::vector<uint32_t> m_circleIndices;

    // SPH particle data
    std::vector<float> m_particlePositions;
    std::vector<uint32_t> m_particleColors;
    float m_particleRadius = 0.1f;

    // SPH class
    SPHSolver* m_solver = nullptr;
};

AnitoWave::AnitoWave(const Config &config) : m_config(config), m_width(config.width), m_height(config.height) {
}

AnitoWave::~AnitoWave() {
    delete m_solver;
    if (m_window) {
        bgfx::shutdown();
        glfwTerminate();
    }
}

void AnitoWave::initParticleRendering() {
    m_particleLayout.begin()
        .add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
        .add(bgfx::Attrib::Color0, 4, bgfx::AttribType::Uint8, true)
        .end();

    generateCircleTemplate(m_particleRadius, 32);

    int particlesPerRow = 70;
    int numParticles = particlesPerRow * particlesPerRow;
    float spacing = m_particleRadius * 2.0f;
    float startX = -((particlesPerRow - 1) * spacing) / 2.0f;
    float startY = 0.5f;

    m_particlePositions.reserve(numParticles * 2);
    m_particleColors.reserve(numParticles);

    for (int y = 0; y < particlesPerRow; ++y) {
        for (int x = 0; x < particlesPerRow; ++x) {
            float px = startX + x * spacing;
            float py = startY + y * spacing;

            m_particlePositions.push_back(px);
            m_particlePositions.push_back(py);

            m_particleColors.push_back(0xffff0000);
        }
    }

    // Init SPH particles

    m_solver = new SPHSolver(numParticles);

    const std::vector initVelocities(numParticles * 2, 0.0f);

    m_program = loadProgram("vs_particles", "fs_particles");

    m_solver->init(m_particlePositions, initVelocities);

    m_particleRadius = m_solver->getParams().particleSize;
}

void AnitoWave::generateCircleTemplate(float radius, int segments) {
    m_circleTemplate.clear();
    m_circleIndices.clear();

    m_circleTemplate.push_back({0.f, 0.f, 0.f, 0xffffffff});

    for (int i = 0; i < segments; ++i) {
        float angle = (float)i / segments * bx::kPi * 2.0f;
        float x = bx::cos(angle) * radius;
        float y = bx::sin(angle) * radius;
        m_circleTemplate.push_back({x, y, 0.f, 0xffffffff});
    }

    for (int i = 0; i < segments; ++i) {
        m_circleIndices.push_back(0);
        m_circleIndices.push_back(i + 1);
        m_circleIndices.push_back((i + 1) % segments + 1);
    }
}

void AnitoWave::setupOrtho() {
    const float aspect = (float)m_width / (float)m_height;
    const float height = 2.0f;
    const float width = height * aspect;
    float proj[16];
    bx::mtxOrtho(proj, -width / 2, width / 2, -height / 2, height / 2, 0.0f, 100.0f, 0.0f, bgfx::getCaps()->homogeneousDepth);
    bgfx::setViewTransform(m_kClearView, nullptr, proj);
}

void AnitoWave::renderParticles() {
    std::vector<PosColorVertex> vertices;
    std::vector<uint32_t> indices;

    const size_t vertsPerCircle = m_circleTemplate.size();
    const size_t indicesPerCircle = m_circleIndices.size();
    const size_t numParticles = m_particlePositions.size() / 2;

    vertices.reserve(vertsPerCircle * numParticles);
    indices.reserve(indicesPerCircle * numParticles);

    for (size_t i = 0; i < numParticles; ++i) {
        float px = m_particlePositions[i * 2];
        float py = m_particlePositions[i * 2 + 1];
        uint32_t color = m_particleColors[i];

        uint32_t baseVertex = vertices.size();

        for (const auto& v : m_circleTemplate) {
            vertices.push_back({
                px + v.x * m_particleRadius,
                py + v.y * m_particleRadius,
                v.z,
                color
            });
        }

        for (uint32_t idx: m_circleIndices) {
            indices.push_back(baseVertex + idx);
        }
    }

    if (vertices.empty()) return;

    bgfx::TransientVertexBuffer tvb{};
    bgfx::TransientIndexBuffer tib{};

    if (bgfx::allocTransientBuffers(&tvb, m_particleLayout, vertices.size(), &tib, indices.size(), true)) {
        bx::memCopy(tvb.data, vertices.data(), vertices.size() * sizeof(PosColorVertex));
        bx::memCopy(tib.data, indices.data(), indices.size() * sizeof(uint32_t));

        bgfx::setVertexBuffer(0, &tvb);
        bgfx::setIndexBuffer(&tib);
        bgfx::setState(BGFX_STATE_WRITE_RGB
             | BGFX_STATE_WRITE_A
             | BGFX_STATE_BLEND_FUNC(BGFX_STATE_BLEND_SRC_ALPHA, BGFX_STATE_BLEND_INV_SRC_ALPHA));
        bgfx::submit(m_kClearView, m_program);
    }
}

void AnitoWave::renderImGui() {
    double mx, my;
    glfwGetCursorPos(m_window, &mx, &my);

    uint8_t mouseButtons = 0;
    if (glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        mouseButtons |= IMGUI_MBUT_LEFT;
    }
    if (glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
        mouseButtons |= IMGUI_MBUT_RIGHT;
    }
    if (glfwGetMouseButton(m_window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS) {
        mouseButtons |= IMGUI_MBUT_MIDDLE;
    }

    imguiBeginFrame(
        (int32_t)mx,  // mouse x
        (int32_t)my,  // mouse y
        mouseButtons,  // mouse buttons
        m_scroll,  // mouse scroll
        m_width,
        m_height
    );

    m_scroll = 0;

    if (m_showParamEditor && m_solver) {
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(300, 400), ImGuiCond_FirstUseEver);

        if (ImGui::Begin("SPH Parameters", &m_showParamEditor)) {
            SPHParams& params = m_solver->getParams();

            ImGui::Text("Particle Properties");
            ImGui::SliderFloat("Particle Size", &params.particleSize, 0.01f, 0.2f);
            ImGui::SliderFloat("Smoothing Radius", &params.smoothingRadius, 0.1f, 1.0f);

            ImGui::Separator();
            ImGui::Text("Forces");
            ImGui::SliderFloat("Gravity", &params.gravity, 0.0f, 20.0f);
            ImGui::SliderFloat("Viscosity", &params.viscosityStrength, 0.0f, 20.0f);
            ImGui::SliderFloat("Target Density", &params.targetDensity, 5.0f, 2000.0f);
            ImGui::SliderFloat("Pressure Multiplier", &params.pressureMultiplier, 10.0f, 2000.0f);
            ImGui::SliderFloat("Near Pressure Multiplier", &params.nearPressureMultiplier, 0.0f, 100.0f);

            ImGui::Separator();
            ImGui::Text("Collision");
            ImGui::SliderFloat("Collision Damping", &params.collisionDamping, 0.0f, 1.0f);
            ImGui::SliderFloat("Bounds X", &params.boundsX, 1.0f, 10.0f);
            ImGui::SliderFloat("Bounds Y", &params.boundsY, 1.0f, 10.0f);

            ImGui::Separator();
            if (ImGui::Button("Reset to Defaults")) {
                params = SPHParams();
            }
        }
        ImGui::End();
    }

    imguiEndFrame();
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
    glfwSetScrollCallback(m_window, glfwScrollCallback);
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

    imguiCreate();

    initParticleRendering();
    setupOrtho();

    return true;
}

void AnitoWave::run() {
    double lastTime = glfwGetTime();

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

        double currentTime = glfwGetTime();
        float dt = (float)(currentTime - lastTime);
        dt = max(dt, 0.016f);
        lastTime = currentTime;

        const int substeps = 3;
        float subDt = dt / substeps;

        // Simulation and rendering
        for (int i = 0; i < substeps; i++) {
            if (m_solver) {
                m_solver->update(subDt);
            }
        }

        m_solver->getPositions(m_particlePositions);

        renderParticles();
        renderImGui();

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

    if (app && key == GLFW_KEY_F2 && action == GLFW_RELEASE) {
        app->m_showParamEditor = !app->m_showParamEditor;
    }
}

void AnitoWave::glfwScrollCallback(GLFWwindow *window, double xoffset, double yoffset) {
    AnitoWave* app = static_cast<AnitoWave *>(glfwGetWindowUserPointer(window));
    if (app) {
        app->m_scroll += (int32_t)yoffset;
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