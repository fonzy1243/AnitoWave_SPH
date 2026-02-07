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

struct PosColorVertex {
    float x, y, z;
    uint32_t abgr;
};

struct ParticleInstance {
    float x, y, z, pad0;
    float r, g, b, a;
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
    static void glfwMouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void glfwCursorPosCallback(GLFWwindow* window, double xpos, double ypos);

    void initParticleRendering();
    void initColliderRendering();
    void renderParticles();
    void renderColliders();
    void renderImGui();
    void generateSphereTemplate(int stacks, int slices);
    void generateCubeTemplate();
    void updateCamera();

    Config m_config;
    GLFWwindow* m_window = nullptr;
    uint32_t m_width;
    uint32_t m_height;
    int32_t m_scroll = 0;
    bool m_showStats = false;
    bool m_showParamEditor = true;
    const bgfx::ViewId m_kClearView = 0;

    // Camera
    float m_cameraDistance = 10.0f;
    float m_cameraYaw = 0.0f;
    float m_cameraPitch = 30.0f;
    float m_cameraTarget[3] = {0.0f, 0.0f, 0.0f};
    bool m_mousePressed = false;
    double m_lastMouseX = 0.0;
    double m_lastMouseY = 0.0;

    // bgfx structs
    bgfx::ProgramHandle m_program;
    bgfx::VertexLayout m_particleLayout;
    bgfx::VertexLayout m_instanceLayout;
    bgfx::UniformHandle m_particleRadiusUniform;

    // Particle rendering data
    bgfx::VertexBufferHandle m_circleVB;
    bgfx::IndexBufferHandle m_circleIB;
    std::vector<PosColorVertex> m_circleTemplate;
    std::vector<uint32_t> m_circleIndices;

    // Cube collider rendering data
    bgfx::VertexBufferHandle m_cubeVB;
    bgfx::IndexBufferHandle m_cubeIB;

    // SPH particle data
    float* m_particlePositions = nullptr;
    std::vector<uint32_t> m_particleColors;
    float m_particleRadius = 0.1f;

    // SPH class
    SPHSolver* m_solver = nullptr;
};

AnitoWave::AnitoWave(const Config &config) : m_config(config), m_width(config.width), m_height(config.height) {
}

AnitoWave::~AnitoWave() {
    delete m_solver;

    if (m_particlePositions) {
        cudaFreeHost(m_particlePositions);
        m_particlePositions = nullptr;
    }

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

    m_instanceLayout.begin()
        .add(bgfx::Attrib::TexCoord7, 4, bgfx::AttribType::Float)
        .add(bgfx::Attrib::TexCoord6, 4, bgfx::AttribType::Float, true)
        .end();

    generateSphereTemplate(12, 32);

    m_circleVB = bgfx::createVertexBuffer(
        bgfx::makeRef(m_circleTemplate.data(), static_cast<uint32_t>(m_circleTemplate.size() * sizeof(PosColorVertex))),
        m_particleLayout
    );

    m_circleIB = bgfx::createIndexBuffer(
        bgfx::makeRef(m_circleIndices.data(), static_cast<uint32_t>(m_circleIndices.size() * sizeof(uint32_t))),
        BGFX_BUFFER_INDEX32
    );

    int particlesPerSide = 50;
    int numParticles = particlesPerSide * particlesPerSide * particlesPerSide;

    float boundsX = 10.0f;
    float boundsY = 10.0f;
    float boundsZ = 10.0f;

    float maxSpacingX = boundsX / particlesPerSide;
    float maxSpacingY = boundsY / particlesPerSide;
    float maxSpacingZ = boundsZ / particlesPerSide;

    float spacing = bx::min(m_particleRadius * 2.0f, bx::min(maxSpacingX, bx::min(maxSpacingY, maxSpacingZ))) * 0.95f;;
    float startX = -((particlesPerSide - 1) * spacing) / 2.0f;
    float startY = -((particlesPerSide - 1) * spacing) / 2.0f;
    float startZ = -((particlesPerSide - 1) * spacing) / 2.0f;

    size_t dataSize = numParticles * 3 * sizeof(float);
    cudaMallocHost((void**)&m_particlePositions, dataSize);

    if (!m_particlePositions) {
        fprintf(stderr, "Failed to allocate pinned memory.\n");
        return;
    }

    m_particleColors.reserve(numParticles);

    for (int z = 0; z < particlesPerSide; ++z) {
        for (int y = 0; y < particlesPerSide; ++y) {
            for (int x = 0; x < particlesPerSide; ++x) {
                float px = startX + x * spacing;
                float py = startY + y * spacing;
                float pz = startZ + z * spacing;

                int index = (z * particlesPerSide * particlesPerSide + y * particlesPerSide + x) * 3;
                m_particlePositions[index + 0] = px;
                m_particlePositions[index + 1] = py;
                m_particlePositions[index + 2] = pz;

                m_particleColors.push_back(0xffff0000);
            }
        }
    }

    // Init SPH particles

    m_solver = new SPHSolver(numParticles);

    const std::vector initVelocities(numParticles * 3, 0.0f);

    m_program = loadProgram("vs_particles", "fs_particles");

    m_particleRadiusUniform = bgfx::createUniform("u_particleRadius", bgfx::UniformType::Vec4);

    std::vector tempPos(m_particlePositions, m_particlePositions + numParticles * 3);
    m_solver->init(tempPos, initVelocities);

    initColliderRendering();

    // Add a Sphere at (0, 0, 0) with radius 1.0
    Collider sphere1{};
    sphere1.type = TYPE_SPHERE;
    sphere1.isDynamic = true;
    sphere1.mass = 5050.0f;
    sphere1.position = make_float3(0.0f, 0.0f, 0.0f);
    sphere1.velocity = make_float3(0.0f, 0.0f, 0.0f);
    sphere1.forceAccumulator = make_float3(0.0f, 0.0f, 0.0f);
    sphere1.dims = make_float3(1.0f, 0.0f, 0.0f); // dims.x is radius

    Collider sphere2{};
    sphere2.type = TYPE_SPHERE;
    sphere2.isDynamic = true;
    sphere2.mass = 1050.0f;
    sphere2.position = make_float3(5.5f, 0.0f, 2.0f);
    sphere2.velocity = make_float3(0.0f, 0.0f, 0.0f);
    sphere2.forceAccumulator = make_float3(0.0f, 0.0f, 0.0f);
    sphere2.dims = make_float3(0.6f, 0.0f, 0.0f); // dims.x is radius

    Collider sphere3{};
    sphere3.type = TYPE_SPHERE;
    sphere3.isDynamic = true;
    sphere3.mass = 2050.0f;
    sphere3.position = make_float3(-5.5f, 0.0f, 2.0f);
    sphere3.velocity = make_float3(0.0f, 0.0f, 0.0f);
    sphere3.forceAccumulator = make_float3(0.0f, 0.0f, 0.0f);
    sphere3.dims = make_float3(0.8f, 0.0f, 0.0f); // dims.x is radius

    // Collider box1{};
    // box1.type = TYPE_BOX;
    // box1.isDynamic = true;
    // box1.mass = 2050.0f;
    // box1.position = make_float3(5.5f, 0.0f, 2.0f);
    // box1.velocity = make_float3(0.0f, 0.0f, 0.0f);
    // box1.forceAccumulator = make_float3(0.0f, 0.0f, 0.0f);
    // box1.dims = make_float3(0.6f, 0.3f, 0.6f); // dims.x is radius

    m_solver->addCollider(sphere1);
    m_solver->addCollider(sphere2);
    m_solver->addCollider(sphere3);
    // m_solver->addCollider(box1);

    m_particleRadius = m_solver->getParams().particleSize;
}

void AnitoWave::initColliderRendering() {
    generateCubeTemplate();
}

void AnitoWave::generateSphereTemplate(int stacks, int slices) {
    m_circleTemplate.clear();
    m_circleIndices.clear();

    for (int i = 0; i <= stacks; ++i) {
        float v = (float)i / (float)stacks;
        float phi = v * bx::kPi;

        for (int j = 0; j <= slices; ++j) {
            float u = (float)j / (float)slices;
            float theta = u * bx::kPi * 2.0f;

            float x = bx::sin(phi) * bx::cos(theta);
            float y = bx::cos(phi);
            float z = bx::sin(phi) * bx::sin(theta);

            m_circleTemplate.push_back({x, y, z, 0xffffffff});
        }
    }

    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            int p1 = i * (slices + 1) + j;
            int p2 = p1 + (slices + 1);

            m_circleIndices.push_back(p1);
            m_circleIndices.push_back(p2);
            m_circleIndices.push_back(p1 + 1);

            m_circleIndices.push_back(p1 + 1);
            m_circleIndices.push_back(p2);
            m_circleIndices.push_back(p2 + 1);
        }
    }
}

void AnitoWave::generateCubeTemplate() {
    PosColorVertex vertices[] = {
        {-1.0f,  1.0f,  1.0f, 0xffffffff}, { 1.0f,  1.0f,  1.0f, 0xffffffff},
        {-1.0f, -1.0f,  1.0f, 0xffffffff}, { 1.0f, -1.0f,  1.0f, 0xffffffff},
        {-1.0f,  1.0f, -1.0f, 0xffffffff}, { 1.0f,  1.0f, -1.0f, 0xffffffff},
        {-1.0f, -1.0f, -1.0f, 0xffffffff}, { 1.0f, -1.0f, -1.0f, 0xffffffff},
    };

    const uint32_t indices[] = {
        0, 1, 2,
        1, 3, 2,
        4, 6, 5,
        5, 6, 7,
        0, 2, 4,
        4, 2, 6,
        1, 5, 3,
        5, 7, 3,
        0, 4, 1,
        2, 3, 6,
        6, 3, 7,
    };

    m_cubeVB = bgfx::createVertexBuffer(
        bgfx::makeRef(vertices, sizeof(vertices)),
        m_particleLayout
    );
    m_cubeIB = bgfx::createIndexBuffer(
        bgfx::makeRef(indices, sizeof(indices))
    );
}

void AnitoWave::updateCamera() {
    const float aspect = (float)m_width / (float)m_height;

    float radYaw = bx::toRad(m_cameraYaw);
    float radPitch = bx::toRad(m_cameraPitch);

    float3 eye = {
        m_cameraTarget[0] + m_cameraDistance * bx::cos(radPitch) * bx::sin(radYaw),
        m_cameraTarget[1] + m_cameraDistance * bx::sin(radPitch),
        m_cameraTarget[2] + m_cameraDistance * bx::cos(radPitch) * bx::cos(radYaw)
    };

    float3 at = {m_cameraTarget[0], m_cameraTarget[1], m_cameraTarget[2]};
    float3 up = {0.0f, 1.0f, 0.0f};

    float view[16];
    float proj[16];

    bx::mtxLookAt(view, {eye.x, eye.y, eye.z}, {at.x, at.y, at.z}, {up.x, up.y, up.z});
    bx::mtxProj(proj, 60.0f, aspect, 0.1f, 100.0f, bgfx::getCaps()->homogeneousDepth);

    bgfx::setViewTransform(m_kClearView, view, proj);
}

void AnitoWave::renderParticles() {
    const size_t numParticles = m_solver->getNumParticles();
    if (numParticles == 0) return;

    std::vector<ParticleInstance> instances;
    instances.reserve(numParticles);

    for (size_t i = 0; i < numParticles; ++i) {
        uint32_t c = m_particleColors[i];
        float a = ((c >> 24) & 0xff) / 255.f;
        float b = ((c >> 16) & 0xff) / 255.f;
        float g = ((c >> 8) & 0xff) / 255.f;
        float r = ((c) & 0xff) / 255.f;

        instances.push_back({
            m_particlePositions[i * 3],
            m_particlePositions[i * 3 + 1],
            m_particlePositions[i * 3 + 2],
            0.0f,
            r, g, b, a
        });
    }

    uint32_t maxAvailable = bgfx::getAvailInstanceDataBuffer(numParticles, m_instanceLayout.getStride());
    const uint32_t maxInstancesPerBatch = bx::min(maxAvailable, 1048576u);

    uint32_t offset = 0;
    while (offset < numParticles) {
        uint32_t count = bx::min(maxInstancesPerBatch, (uint32_t)numParticles - offset);

        bgfx::InstanceDataBuffer idb{};
        bgfx::allocInstanceDataBuffer(&idb, count, m_instanceLayout.getStride());

        if (idb.data != NULL) {
            memcpy(idb.data, &instances[offset], count * sizeof(ParticleInstance));

            bgfx::setVertexBuffer(0, m_circleVB);
            bgfx::setIndexBuffer(m_circleIB);
            bgfx::setInstanceDataBuffer(&idb);

            float radiusData[4] = { m_particleRadius, 0.0f, 0.0f, 0.0f };
            bgfx::setUniform(m_particleRadiusUniform, radiusData);

            bgfx::setState(BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A | BGFX_STATE_WRITE_Z
                | BGFX_STATE_DEPTH_TEST_LESS | BGFX_STATE_BLEND_FUNC(BGFX_STATE_BLEND_SRC_ALPHA, BGFX_STATE_BLEND_INV_SRC_ALPHA));

            bgfx::submit(m_kClearView, m_program);

            offset += count;
        } else {
            fprintf(stderr, "Instance buffer allocation failed.\n");
        }
    }
}

void AnitoWave::renderColliders() {
    if (!m_solver) return;

    std::vector<Collider> colliders;
    m_solver->getColliders(colliders);

    float defaultRadius[4] = { 1.0f, 0.0f, 0.0f, 0.0f };
    bgfx::setUniform(m_particleRadiusUniform, defaultRadius);

    bgfx::InstanceDataBuffer idb{};
    if (bgfx::getAvailInstanceDataBuffer(1, m_instanceLayout.getStride()) > 0) {
        bgfx::allocInstanceDataBuffer(&idb, 1, m_instanceLayout.getStride());
        ParticleInstance* data = (ParticleInstance*)idb.data;
        data[0].x = 0.0f; data[0].y = 0.0f; data[0].z = 0.0f; data[0].pad0 = 0.0f;
        data[0].r = 1.0f; data[0].g = 1.0f; data[0].b = 1.0f; data[0].a = 1.0f;
    }

    for (const auto& col : colliders) {
        float mtx[16];

        float mtxTrans[16];
        bx::mtxTranslate(mtxTrans, col.position.x, col.position.y, col.position.z);

        float mtxScale[16];
        if (col.type == TYPE_SPHERE) {
            bx::mtxScale(mtxScale, col.dims.x, col.dims.x, col.dims.x);

            bx::mtxMul(mtx, mtxScale, mtxTrans);

            bgfx::setVertexBuffer(0, m_circleVB);
            bgfx::setIndexBuffer(m_circleIB);
        } else if (col.type == TYPE_BOX) {
            bx::mtxScale(mtxScale, col.dims.x, col.dims.y, col.dims.z);

            bx::mtxMul(mtx, mtxScale, mtxTrans);

            bgfx::setVertexBuffer(0, m_cubeVB);
            bgfx::setIndexBuffer(m_cubeIB);
        }

        bgfx::setTransform(mtx);

        bgfx::setInstanceDataBuffer(&idb);

        bgfx::setState(BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A | BGFX_STATE_WRITE_Z | BGFX_STATE_DEPTH_TEST_LESS);

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
            ImGui::SliderFloat("Bounds X", &params.boundsX, 1.0f, 30.0f);
            ImGui::SliderFloat("Bounds Y", &params.boundsY, 1.0f, 30.0f);
            ImGui::SliderFloat("Bounds Z", &params.boundsZ, 1.0f, 50.0f);

            ImGui::Separator();
            ImGui::Text("Colliders");
            ImGui::SliderFloat("Collider Drag", &params.colliderDragMultiplier, 0.0f, 0.5f);

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
    glfwSetMouseButtonCallback(m_window, glfwMouseButtonCallback);
    glfwSetCursorPosCallback(m_window, glfwCursorPosCallback);
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
    bgfx::setViewClear(m_kClearView, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, 0x303030ff, 1.0f, 0);
    bgfx::setViewRect(m_kClearView, 0, 0, bgfx::BackbufferRatio::Equal);

    imguiCreate();

    initParticleRendering();
    updateCamera();

    return true;
}

void AnitoWave::run() {
    double lastTime = glfwGetTime();
    double accumulator = 0.0f;

    const float FIXED_DT = 1.0f / 120.0f;
    const int MAX_STEPS_PER_FRAME = 3;

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
        double frameTime = currentTime - lastTime;
        lastTime = currentTime;

        if (frameTime > 0.25) frameTime = 0.25;

        accumulator += frameTime;

        int steps = 0;

        // Simulation and rendering
        while (accumulator >= FIXED_DT && steps < MAX_STEPS_PER_FRAME) {
            if (m_solver) {
                m_solver->update(FIXED_DT);
            }
            accumulator -= FIXED_DT;
            steps++;
        }

        if (accumulator > FIXED_DT) {
            accumulator = 0.0f;
        }

        m_solver->getPositions(m_particlePositions);

        renderParticles();
        renderColliders();
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
        app->m_cameraDistance -= (float)yoffset * 0.5f;
        app->m_cameraDistance = bx::clamp(app->m_cameraDistance, 2.0f, 50.0f);
        app->updateCamera();
    }
}

void AnitoWave::glfwMouseButtonCallback(GLFWwindow *window, int button, int action, int mods) {
    AnitoWave* app = static_cast<AnitoWave *>(glfwGetWindowUserPointer(window));
    if (app && button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            app->m_mousePressed = true;
        } else if (action == GLFW_RELEASE) {
            app->m_mousePressed = false;
        }
    }
}

void AnitoWave::glfwCursorPosCallback(GLFWwindow *window, double xpos, double ypos) {
    AnitoWave* app = static_cast<AnitoWave *>(glfwGetWindowUserPointer(window));
    if (app && app->m_mousePressed) {
        double dx = xpos - app->m_lastMouseX;
        double dy = ypos - app->m_lastMouseY;

        app->m_cameraYaw += (float)dx * 0.5f;
        app->m_cameraPitch += (float)dy * 0.5f;

        app->m_cameraPitch = bx::clamp(app->m_cameraPitch, -89.0f, 89.0f);

        app->m_lastMouseX = xpos;
        app->m_lastMouseY = ypos;

        app->updateCamera();
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