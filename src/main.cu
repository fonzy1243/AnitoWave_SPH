#include <cstdio>
#include <vector>
#include <immintrin.h>
#include <execution>
#include <algorithm>
#include <numeric>
#include <cstdint>
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

#define TINYLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

struct ParticleInstance {
    float x, y, z, pad0;
    float vx, vy, vz, pad1;
};

void processGLTFNode(const tinygltf::Model& model, int nodeIndex, const float* parentMatrix,
                     std::vector<std::array<double, 3>>& outVertices,
                     std::vector<std::array<int, 3>>& outTriangles,
                     std::vector<PosColorVertex>& outRenderVertices,
                     std::vector<uint32_t>& outRenderIndices,
                     float scale)
{
    const tinygltf::Node& node = model.nodes[nodeIndex];
    float localMatrix[16];
    bx::mtxIdentity(localMatrix);

    if (node.matrix.size() == 16) {
        for (int i = 0; i < 16; ++i) localMatrix[i] = static_cast<float>(node.matrix[i]);
    } else {
        float t[16], r[16], s[16];
        bx::mtxIdentity(t); bx::mtxIdentity(r); bx::mtxIdentity(s);

        if (node.translation.size() == 3) {
            bx::mtxTranslate(t, (float)node.translation[0], (float)node.translation[1], (float)node.translation[2]);
        }
        if (node.rotation.size() == 4) {
            bx::Quaternion quat = {
                (float)node.rotation[0],
                (float)node.rotation[1],
                (float)node.rotation[2],
                (float)node.rotation[3]
            };

            bx::mtxFromQuaternion(r, quat);
        }
        if (node.scale.size() == 3) {
            bx::mtxScale(s, (float)node.scale[0], (float)node.scale[1], (float)node.scale[2]);
        }

        // TRS: Scale -> Rotate -> Translate
        float temp[16];
        bx::mtxMul(temp, s, r);
        bx::mtxMul(localMatrix, temp, t);
    }

    float globalMatrix[16];
    bx::mtxMul(globalMatrix, localMatrix, parentMatrix);

    if (node.mesh >= 0) {
        const tinygltf::Mesh& mesh = model.meshes[node.mesh];
        for (const auto& primitive : mesh.primitives) {

            const tinygltf::Accessor& posAccessor = model.accessors[primitive.attributes.at("POSITION")];
            const tinygltf::BufferView& posView = model.bufferViews[posAccessor.bufferView];
            const tinygltf::Buffer& posBuffer = model.buffers[posView.buffer];
            const float* positions = reinterpret_cast<const float*>(&posBuffer.data[posView.byteOffset + posAccessor.byteOffset]);

            const float* normals = nullptr;
            if (primitive.attributes.find("NORMAL") != primitive.attributes.end()) {
                const tinygltf::Accessor& normAccessor = model.accessors[primitive.attributes.at("NORMAL")];
                const tinygltf::BufferView& normView = model.bufferViews[normAccessor.bufferView];
                const tinygltf::Buffer& normBuffer = model.buffers[normView.buffer];
                normals = reinterpret_cast<const float*>(&normBuffer.data[normView.byteOffset + normAccessor.byteOffset]);
            }

            uint32_t vertexOffset = outVertices.size();

            for (size_t i = 0; i < posAccessor.count; ++i) {
                float px = positions[i*3+0] * scale;
                float py = positions[i*3+1] * scale;
                float pz = positions[i*3+2] * scale;

                float vx = px * globalMatrix[0] + py * globalMatrix[4] + pz * globalMatrix[8] + globalMatrix[12];
                float vy = px * globalMatrix[1] + py * globalMatrix[5] + pz * globalMatrix[9] + globalMatrix[13];
                float vz = px * globalMatrix[2] + py * globalMatrix[6] + pz * globalMatrix[10] + globalMatrix[14];

                outVertices.push_back({(double)vx, (double)vy, (double)vz});

                PosColorVertex rv = {vx, vy, vz, 0,0,0, 0xffaaaaaa};

                if (normals) {
                    float nx = normals[i*3+0];
                    float ny = normals[i*3+1];
                    float nz = normals[i*3+2];

                    rv.nx = nx * globalMatrix[0] + ny * globalMatrix[4] + nz * globalMatrix[8];
                    rv.ny = nx * globalMatrix[1] + ny * globalMatrix[5] + nz * globalMatrix[9];
                    rv.nz = nx * globalMatrix[2] + ny * globalMatrix[6] + nz * globalMatrix[10];

                    // normalize
                    float len = std::sqrt(rv.nx*rv.nx + rv.ny*rv.ny + rv.nz*rv.nz);
                    if (len > 0.0f) {
                        rv.nx /= len; rv.ny /= len; rv.nz /= len;
                    }
                }
                outRenderVertices.push_back(rv);
            }

            // --- Extract Indices ---
            if (primitive.indices >= 0) {
                const tinygltf::Accessor& indAccessor = model.accessors[primitive.indices];
                const tinygltf::BufferView& indView = model.bufferViews[indAccessor.bufferView];
                const tinygltf::Buffer& indBuffer = model.buffers[indView.buffer];
                const uint8_t* indexData = &indBuffer.data[indView.byteOffset + indAccessor.byteOffset];

                for (size_t i = 0; i < indAccessor.count; i += 3) {
                    int i0, i1, i2;
                    if (indAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                        const uint16_t* ind = reinterpret_cast<const uint16_t*>(indexData);
                        i0 = ind[i+0]; i1 = ind[i+1]; i2 = ind[i+2];
                    } else if (indAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                        const uint32_t* ind = reinterpret_cast<const uint32_t*>(indexData);
                        i0 = ind[i+0]; i1 = ind[i+1]; i2 = ind[i+2];
                    } else if (indAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                        const uint8_t* ind = indexData;
                        i0 = ind[i+0]; i1 = ind[i+1]; i2 = ind[i+2];
                    } else {
                        continue;
                    }
                    int vOffset = static_cast<int>(vertexOffset);
                    outTriangles.push_back({vOffset + i0, vOffset + i1, vOffset + i2});
                    outRenderIndices.push_back(vertexOffset + i0); outRenderIndices.push_back(vertexOffset + i1); outRenderIndices.push_back(vertexOffset + i2);
                }
            } else {
                for (size_t i = 0; i < posAccessor.count; i += 3) {
                    int vOffset = static_cast<int>(vertexOffset);
                    outTriangles.push_back({vOffset + (int)i, vOffset + (int)i + 1, vOffset + (int)i + 2});
                    outRenderIndices.push_back(vertexOffset + (uint32_t)i); outRenderIndices.push_back(vertexOffset + (uint32_t)i + 1); outRenderIndices.push_back(vertexOffset + (uint32_t)i + 2);
                }
            }
        }
    }

    for (int childIndex : node.children) {
        processGLTFNode(model, childIndex, globalMatrix, outVertices, outTriangles, outRenderVertices, outRenderIndices, scale);
    }
}

bool loadMeshRawData(const std::string& filepath,
                     std::vector<std::array<double, 3>>& outVertices,
                     std::vector<std::array<int, 3>>& outTriangles,
                     std::vector<PosColorVertex>& outRenderVertices,
                     std::vector<uint32_t>& outRenderIndices,
                     float scale = 1.0f) {
    std::string ext = filepath.substr(filepath.find_last_of('.') + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    // Parse .obj
    if (ext == "obj") {
        tinyobj::ObjReaderConfig reader_config;
        reader_config.triangulate = true;
        tinyobj::ObjReader reader;

        if (!reader.ParseFromFile(filepath, reader_config)) {
            std::cerr << "TinyObjReader Error: " << reader.Error() << "\n";
            return false;
        }
        if (!reader.Warning().empty()) {
            std::cerr << "TinyObjReader Warning: " << reader.Warning() << "\n";
        }

        auto& attrib = reader.GetAttrib();
        auto& shapes = reader.GetShapes();
        uint32_t indexOffset = 0;

        for (size_t v = 0; v < attrib.vertices.size() / 3; v++) {
            outVertices.push_back({
                attrib.vertices[3*v+0] * scale,
                attrib.vertices[3*v+1] * scale,
                attrib.vertices[3*v+2] * scale
            });
            PosColorVertex rv = {
                attrib.vertices[3*v+0] * scale,
                attrib.vertices[3*v+1] * scale,
                attrib.vertices[3*v+2] * scale,
                0,0,0, 0xffaaaaaa
            };

            if (attrib.normals.size() > 3 * v + 2) {
                rv.nx = attrib.normals[3*v+0]; rv.ny = attrib.normals[3*v+1]; rv.nz = attrib.normals[3*v+2];
            }
            outRenderVertices.push_back(rv);
        }

        for (const auto& shape : shapes) {
            for (size_t i = 0; i < shape.mesh.indices.size(); i += 3) {
                int i0 = shape.mesh.indices[i+0].vertex_index;
                int i1 = shape.mesh.indices[i+1].vertex_index;
                int i2 = shape.mesh.indices[i+2].vertex_index;

                outTriangles.push_back({i0, i1, i2});
                outRenderIndices.push_back(i0); outRenderIndices.push_back(i1); outRenderIndices.push_back(i2);
            }
        }
        return true;
    }
    if (ext == "gltf" || ext == "glb") {
        tinygltf::Model model;
        tinygltf::TinyGLTF loader;
        std::string err, warn;

        bool ret = (ext == "glb") ? loader.LoadBinaryFromFile(&model, &err, &warn, filepath)
                                  : loader.LoadASCIIFromFile(&model, &err, &warn, filepath);

        if (!warn.empty()) std::cout << "GLTF Warn: " << warn << "\n";
        if (!err.empty()) std::cerr << "GLTF Error: " << err << "\n";
        if (!ret) return false;

        // Iterate through all meshes and primitives in the GLTF
        int sceneIndex = model.defaultScene > -1 ? model.defaultScene : 0;
        const tinygltf::Scene& scene = model.scenes[sceneIndex];

        float identity[16];
        bx::mtxIdentity(identity);

        for (int nodeIndex : scene.nodes)
        {
            processGLTFNode(model, nodeIndex, identity, outVertices, outTriangles, outRenderVertices, outRenderIndices, scale);
        }
        return true;
    }
}

static void precompute_coords_avx512(double* __restrict__ out, double minVal, double maxVal, int resolution) noexcept
{
    const double scale = (maxVal - minVal) / (resolution - 1);

    __m512d vmin = _mm512_set1_pd(minVal);
    __m512d vscale = _mm512_set1_pd(scale);

    __m512d vbase = _mm512_set_pd(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    __m512d vstep = _mm512_set1_pd(8.0);

    int i = 0;
    for (; i + 8 <= resolution; i += 8)
    {
        __m512d result = _mm512_fmadd_pd(vbase, vscale, vmin);
        _mm512_storeu_pd(out + i, result);
        vbase = _mm512_add_pd(vbase, vstep);
    }

    for (; i < resolution; ++i)
    {
        out[i] = minVal + i * scale;
    }
}

void computeDistanceField(std::vector<float>& distanceData, const auto& mesh_distance,
    double minX, double maxX,
    double minY, double maxY,
    double minZ, double maxZ,
    int resolution)
{
    const int N = resolution;
    const int N2 = N * N;
    const int total = N * N * N;

    distanceData.resize(total);

    std::vector<double> xs(N), ys(N), zs(N);

    precompute_coords_avx512(xs.data(), minX, maxX, N);
    precompute_coords_avx512(ys.data(), minY, maxY, N);
    precompute_coords_avx512(zs.data(), minZ, maxZ, N);

    std::vector<int> z_indices(N);
    std::iota(z_indices.begin(), z_indices.end(), 0);

    std::for_each(std::execution::par_unseq, z_indices.begin(), z_indices.end(), [&](int z)
    {
        const double pz = zs[z];
        const int z_off = z * N2;

        for (int y = 0; y < N; ++y)
        {
            const double py = ys[y];
            const int y_off = z_off + y * N;

            for (int x = 0; x < N; ++x)
            {
                tmd::Result dist_result = mesh_distance.signed_distance({xs[x], py, pz});

                distanceData[y_off + x] = static_cast<float>(dist_result.distance);
            }
        }
    });
}

MeshSDFData generateSDFFromMesh(const std::string& filepath, int resolution = 64, float scale = 1.0f) {
    MeshSDFData result = {};

    std::vector<std::array<double, 3>> vertices;
    std::vector<std::array<int, 3>> triangles;

    if (!loadMeshRawData(filepath, vertices, triangles, result.renderVertices, result.renderIndices, scale)) {
        return result;
    }

    std::cout << "Loaded mesh: " << vertices.size() << " vertices, " << triangles.size() << " triangles\n";

    double minX = vertices[0][0], minY = vertices[0][1], minZ = vertices[0][2];
    double maxX = vertices[0][0], maxY = vertices[0][1], maxZ = vertices[0][2];

    for (const auto& v : vertices) {
        minX = std::min(minX, v[0]); minY = std::min(minY, v[1]); minZ = std::min(minZ, v[2]);
        maxX = std::max(maxX, v[0]); maxY = std::max(maxY, v[1]); maxZ = std::max(maxZ, v[2]);
    }

    double padding = 1.0;
    minX -= padding; minY -= padding; minZ -= padding;
    maxX += padding; maxY += padding; maxZ += padding;

    result.minBounds = make_float3(static_cast<float>(minX), static_cast<float>(minY), static_cast<float>(minZ));
    result.maxBounds = make_float3(static_cast<float>(maxX), static_cast<float>(maxY), static_cast<float>(maxZ));
    result.resX = resolution; result.resY = resolution; result.resZ = resolution;

    std::cout << "Generating SDF Voxel Grid...\n";
    tmd::TriangleMeshDistance mesh_distance(vertices, triangles);

    computeDistanceField(result.distanceData, mesh_distance, minX, maxX, minY, maxY, minZ, maxZ, resolution);

    std::cout << "SDF Generation complete.\n";
    return result;
}

void createSDFTextureObject(const MeshSDFData& sdfData, cudaTextureObject_t& outTex, cudaArray_t& outArray) {
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent extent = make_cudaExtent(sdfData.resX, sdfData.resY, sdfData.resZ);

    cudaMalloc3DArray(&outArray, &channelDesc, extent);

    cudaMemcpy3DParms copyParams = {};
    copyParams.srcPtr = make_cudaPitchedPtr(
        (void*)sdfData.distanceData.data(),
        sdfData.resX * sizeof(float),
        sdfData.resX, sdfData.resY
    );
    copyParams.dstArray = outArray;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = outArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1; // sample using [0.0, 1.0] UVW coords

    cudaCreateTextureObject(&outTex, &resDesc, &texDesc, nullptr);
}

struct SceneDef
{
    const char* name;

    const char* terrainMeshPath;
    float terrainScale;
    int terrainSDFResolution;

    float boundsX, boundsY, boundsZ;

    int particlesPerSide;
    float spawnOffsetX, spawnOffsetY, spawnOffsetZ;

    float gravity;
    float targetDensity;
    float pressureMultiplier;
    float viscosityStrength;

    std::vector<Collider> extraColliders;
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

    void initStaticRenderResources();
    void initColliderRendering();
    void renderParticles(bgfx::ViewId viewId, bgfx::ProgramHandle program, uint64_t renderState);
    void renderFluidPasses();
    void renderColliders();
    void renderImGui();
    void generateSphereTemplate(int stacks, int slices);
    void drawFullscreenQuad(bgfx::ViewId viewId, bgfx::ProgramHandle program, uint64_t renderState = BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A);
    void renderFluidComposite();
    void generateCubeTemplate();
    void updateCamera();

    // Scene methods
    void buildSceneList();
    void loadScene(int index);
    void unloadCurrentScene();

    Config m_config;
    GLFWwindow* m_window = nullptr;
    uint32_t m_width;
    uint32_t m_height;
    int32_t m_scroll = 0;
    bool m_showStats = false;
    bool m_showParamEditor = true;
    bool m_showRenderEditor = true;
    const bgfx::ViewId m_kClearView = 0;
    int m_activeScene = 0;

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

    // Mesh rendering data
    bgfx::VertexBufferHandle m_meshVB;
    bgfx::IndexBufferHandle m_meshIB;

    // Per-scene bgfx handles
    bgfx::VertexBufferHandle m_terrainVB = BGFX_INVALID_HANDLE;
    bgfx::IndexBufferHandle m_terrainIB = BGFX_INVALID_HANDLE;

    // Depth handles
    bgfx::TextureHandle m_depthTexture = BGFX_INVALID_HANDLE;
    bgfx::TextureHandle m_hwDepthTexture = BGFX_INVALID_HANDLE;
    bgfx::FrameBufferHandle m_depthFbo = BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle m_depthProgram = BGFX_INVALID_HANDLE;
    const bgfx::ViewId m_kDepthPassView = 1;

    // Depth program test
    bgfx::UniformHandle m_depthSampler;
    bgfx::ProgramHandle m_debugDepthProgram = BGFX_INVALID_HANDLE;

    // Gaussian blur
    float m_fluidBlurSmoothness = 1.0f;
    float m_fluidBlurSize = 10.0f;
    float m_fluidDepthFactor = 1.0f;
    const bgfx::ViewId m_kBlurXPassView = 2;
    const bgfx::ViewId m_kBlurYPassView = 3;
    const bgfx::ViewId m_kCompositePassView = 5;

    bgfx::ProgramHandle m_blurProgram = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle m_blurParamsUniform = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle m_blurFalloffUniform = BGFX_INVALID_HANDLE;

    // Framebuffers for blur ping-pong
    bgfx::TextureHandle m_blurXTexture = BGFX_INVALID_HANDLE;
    bgfx::FrameBufferHandle m_blurXFbo = BGFX_INVALID_HANDLE;

    bgfx::TextureHandle m_blurYTexture = BGFX_INVALID_HANDLE;
    bgfx::FrameBufferHandle m_blurYFbo = BGFX_INVALID_HANDLE;

    // Normals shader
    bgfx::UniformHandle m_texelSizeUniform = BGFX_INVALID_HANDLE;
    bgfx::ProgramHandle m_compositeProgram = BGFX_INVALID_HANDLE;

    // Thickness
    bgfx::ViewId m_kThicknessPassView = 4;
    bgfx::ProgramHandle m_thicknessProgram = BGFX_INVALID_HANDLE;
    bgfx::TextureHandle m_thicknessTexture = BGFX_INVALID_HANDLE;
    bgfx::FrameBufferHandle m_thicknessFrameBuffer = BGFX_INVALID_HANDLE;
    bgfx::UniformHandle m_thicknessSampler = BGFX_INVALID_HANDLE;

    // SPH particle data
    float* m_particlePositions = nullptr;
    float* m_particleVelocities = nullptr;
    std::vector<uint32_t> m_particleColors;
    float m_particleRadius = 0.1f;

    // Scene list
    std::vector<SceneDef> m_scenes;

    // SPH class
    SPHSolver* m_solver = nullptr;
};

void AnitoWave::buildSceneList()
{
    {
        SceneDef empty{};
        empty.name = "Dam Break";
        empty.terrainMeshPath = "";
        empty.terrainScale = 1.0f;
        empty.terrainSDFResolution = 64;

        empty.boundsX = 40.0f; empty.boundsY = 25.0; empty.boundsZ = 12.0f;
        empty.particlesPerSide = 100;

        empty.spawnOffsetX = -14.0f;
        empty.spawnOffsetY = -8.0f;
        empty.spawnOffsetZ = 0.0f;

        empty.gravity = 10;
        empty.targetDensity = 650.0f;
        empty.pressureMultiplier = 550.0f;
        empty.viscosityStrength = 10.00f;

        m_scenes.push_back(empty);
    }

    {
        SceneDef calcata{};
        calcata.name = "Calcata";
        calcata.terrainMeshPath = "meshes/calcata.glb";
        calcata.terrainScale = 1.0f;
        calcata.terrainSDFResolution = 64;
        calcata.boundsX = 15.0f; calcata.boundsY = 50.0f; calcata.boundsZ = 15.0f;
        calcata.particlesPerSide = 50;
        calcata.spawnOffsetY = 15.0f;
        calcata.gravity = 0.f;
        calcata.targetDensity = 650.0f;
        calcata.pressureMultiplier = 150.0f;
        calcata.viscosityStrength = 1.0f;
        m_scenes.push_back(calcata);
    }

    {
        SceneDef mountains{};
        mountains.name = "Mountains";
        mountains.terrainMeshPath = "meshes/Mountains.glb";
        mountains.terrainScale = 10.0f;
        mountains.terrainSDFResolution = 108;
        mountains.boundsX = 15.0f; mountains.boundsY = 50.0f; mountains.boundsZ = 15.0f;
        mountains.particlesPerSide = 50;
        mountains.spawnOffsetY = 15.0f;
        mountains.gravity = 0.f;
        mountains.targetDensity = 650.0f;
        mountains.pressureMultiplier = 150.0f;
        mountains.viscosityStrength = 1.0f;
        m_scenes.push_back(mountains);
    }

    {
        SceneDef hohenzollern{};
        hohenzollern.name = "Hohenzollern";
        hohenzollern.terrainMeshPath = "meshes/Hohenzollern.glb";
        hohenzollern.terrainScale = 1.0f;
        hohenzollern.terrainSDFResolution = 64;
        hohenzollern.boundsX = 15.0f; hohenzollern.boundsY = 50.0f; hohenzollern.boundsZ = 15.0f;
        hohenzollern.particlesPerSide = 50;
        hohenzollern.spawnOffsetY = 15.0f;
        hohenzollern.gravity = 0.f;
        hohenzollern.targetDensity = 650.0f;
        hohenzollern.pressureMultiplier = 150.0f;
        hohenzollern.viscosityStrength = 1.0f;
        m_scenes.push_back(hohenzollern);
    }

    {
        SceneDef sponza{};
        sponza.name = "Sponza";
        sponza.terrainMeshPath = "meshes/sponza/Sponza.gltf";
        sponza.terrainScale = 1.0f;
        sponza.terrainSDFResolution = 64;
        sponza.boundsX = 15.0f; sponza.boundsY = 70.0f; sponza.boundsZ = 15.0f;
        sponza.particlesPerSide = 50;
        sponza.spawnOffsetY = 25.0f;
        sponza.gravity = 0.f;
        sponza.targetDensity = 650.0f;
        sponza.pressureMultiplier = 150.0f;
        sponza.viscosityStrength = 1.0f;
        m_scenes.push_back(sponza);
    }

    {
        SceneDef virtual_city{};
        virtual_city.name = "Virtual City";
        virtual_city.terrainMeshPath = "meshes/VirtualCity.glb";
        virtual_city.terrainScale = 1.0f;
        virtual_city.terrainSDFResolution = 64;
        virtual_city.boundsX = 15.0f; virtual_city.boundsY = 70.0f; virtual_city.boundsZ = 15.0f;
        virtual_city.particlesPerSide = 50;
        virtual_city.spawnOffsetY = 25.0f;
        virtual_city.gravity = 0.f;
        virtual_city.targetDensity = 650.0f;
        virtual_city.pressureMultiplier = 150.0f;
        virtual_city.viscosityStrength = 1.0f;
        m_scenes.push_back(virtual_city);
    }

    {
        SceneDef alien{};
        alien.name = "Alien Terrain";
        alien.terrainMeshPath = "meshes/alien.glb";
        alien.terrainScale = 10.0f;
        alien.terrainSDFResolution = 64;
        alien.boundsX = 15.0f; alien.boundsY = 70.0f; alien.boundsZ = 15.0f;
        alien.particlesPerSide = 50;
        alien.spawnOffsetY = 15.0f;
        alien.gravity = 0.f;
        alien.targetDensity = 650.0f;
        alien.pressureMultiplier = 150.0f;
        alien.viscosityStrength = 1.0f;
        m_scenes.push_back(alien);
    }

    // {
    //     SceneDef abg{};
    //     abg.name = "A Beautiful Game";
    //     abg.terrainMeshPath = "meshes/ABeautifulGame.glb";
    //     abg.terrainScale = 1.0f;
    //     abg.terrainSDFResolution = 64;
    //     abg.boundsX = 15.0f; abg.boundsY = 50.0f; abg.boundsZ = 15.0f;
    //     abg.particlesPerSide = 50;
    //     abg.spawnOffsetY = 15.0f;
    //     abg.gravity = 0.f;
    //     abg.targetDensity = 650.0f;
    //     abg.pressureMultiplier = 150.0f;
    //     abg.viscosityStrength = 1.0f;
    //     m_scenes.push_back(abg);
    // }
}

void AnitoWave::unloadCurrentScene()
{
    delete m_solver;
    m_solver = nullptr;

    if (bgfx::isValid(m_terrainVB)) { bgfx::destroy(m_terrainVB); m_terrainVB = BGFX_INVALID_HANDLE; }
    if (bgfx::isValid(m_terrainIB)) { bgfx::destroy(m_terrainIB); m_terrainIB = BGFX_INVALID_HANDLE; }

    if (m_particlePositions) { cudaFreeHost(m_particlePositions); m_particlePositions = nullptr; }
    if (m_particleVelocities) { cudaFreeHost(m_particleVelocities); m_particleVelocities = nullptr; }
}

void AnitoWave::loadScene(int index)
{
    if (index < 0 || index >= (int)m_scenes.size())
    {
        fprintf(stderr, "loadScene: index %d out of range\n", index);
        return;
    }

    unloadCurrentScene();

    const SceneDef& scene = m_scenes[index];
    std::cout << "Loading scene " << scene.name << std::endl;

    bool hasTerrain = (scene.terrainMeshPath != nullptr && scene.terrainMeshPath[0] != '\0');

    MeshSDFData terrainData;
    cudaTextureObject_t terrainTex = 0;
    cudaArray_t terrainArr = nullptr;

    if (hasTerrain)
    {
        terrainData = generateSDFFromMesh(scene.terrainMeshPath, scene.terrainSDFResolution, scene.terrainScale);

        if (terrainData.distanceData.empty())
        {
            fprintf(stderr, "loadScene: terrain SDF generation failed for '%s'\n", scene.name);
            return;
        }

        m_terrainVB = bgfx::createVertexBuffer(
            bgfx::copy(terrainData.renderVertices.data(), terrainData.renderVertices.size() * sizeof(PosColorVertex)),
            m_particleLayout
        );

        m_terrainIB = bgfx::createIndexBuffer(
            bgfx::copy(terrainData.renderIndices.data(),terrainData.renderIndices.size() * sizeof(uint32_t)),
            BGFX_BUFFER_INDEX32
        );

        createSDFTextureObject(terrainData, terrainTex, terrainArr);
    }

    const int N = scene.particlesPerSide;

    const float spawnFraction = 0.95f;
    const float spawnRangeX   = scene.boundsX * spawnFraction;
    const float spawnRangeY   = scene.boundsY * spawnFraction;
    const float spawnRangeZ   = scene.boundsZ * spawnFraction;

    const float spacing = bx::min(spawnRangeX / N,
                      bx::min(spawnRangeY / N,
                              spawnRangeZ / N)) * 1.05f;

    const float startX = scene.spawnOffsetX - ((N - 1) * spacing) * 0.5f;
    const float startY = scene.spawnOffsetY - ((N - 1) * spacing) * 0.5f;
    const float startZ = scene.spawnOffsetZ - ((N - 1) * spacing) * 0.5f;

    std::vector<float> tempPositions;
    std::vector<uint32_t> tempColors;

    for (int pz = 0; pz < N; ++pz) {
        for (int py = 0; py < N; ++py) {
            for (int px = 0; px < N; ++px) {
                float jitterX = ((rand() % 100) / 100.0f - 0.5f) * 0.02f;
                float jitterY = ((rand() % 100) / 100.0f - 0.5f) * 0.02f;
                float jitterZ = ((rand() % 100) / 100.0f - 0.5f) * 0.02f;

                tempPositions.push_back(startX + px * spacing + jitterX);
                tempPositions.push_back(startY + py * spacing + jitterY);
                tempPositions.push_back(startZ + pz * spacing + jitterZ);
                tempColors.push_back(0xffff0000); // Red fluid
            }
        }
    }

    int totalParticles = tempPositions.size() / 3;

    const size_t posByteSize = totalParticles * 3 * sizeof(float);
    if (cudaMallocHost((void**)&m_particlePositions, posByteSize) != cudaSuccess) {
        fprintf(stderr, "loadScene: cudaMallocHost failed\n");
        return;
    }
    memcpy(m_particlePositions, tempPositions.data(), posByteSize);

    if (cudaMallocHost((void**)&m_particleVelocities, posByteSize) != cudaSuccess)
    {
        fprintf(stderr, "loadScene: cudaMallocHost failed for velocities\n");
        return;
    }
    memset(m_particleVelocities, 0, posByteSize);

    m_particleColors = tempColors;

    m_solver = new SPHSolver(totalParticles);
    {
        SPHParams p;
        p.boundsX = scene.boundsX;
        p.boundsY = scene.boundsY;
        p.boundsZ = scene.boundsZ;
        p.gravity = scene.gravity;
        p.targetDensity = scene.targetDensity;
        p.pressureMultiplier = scene.pressureMultiplier;
        p.viscosityStrength = scene.viscosityStrength;
        m_solver->setParams(p);
    }

    std::vector<float> initVelocities(totalParticles * 3, 0.0f);
    m_solver->init(tempPositions, initVelocities);

    if (hasTerrain) {
        Collider terrainCol{};
        terrainCol.type             = TYPE_MESH;
        terrainCol.isDynamic        = false;
        terrainCol.mass             = 0.0f;
        terrainCol.position         = make_float3(0.0f, 0.0f, 0.0f);
        terrainCol.velocity         = make_float3(0.0f, 0.0f, 0.0f);
        terrainCol.forceAccumulator = make_float3(0.0f, 0.0f, 0.0f);
        terrainCol.sdfTexture       = terrainTex;
        terrainCol.sdfArray         = terrainArr;
        terrainCol.gridMinBounds    = terrainData.minBounds;
        terrainCol.gridMaxBounds    = terrainData.maxBounds;
        terrainCol.voxelSize        = make_float3(
            (terrainData.maxBounds.x - terrainData.minBounds.x) / (terrainData.resX - 1),
            (terrainData.maxBounds.y - terrainData.minBounds.y) / (terrainData.resY - 1),
            (terrainData.maxBounds.z - terrainData.minBounds.z) / (terrainData.resZ - 1));
        m_solver->addCollider(terrainCol);
    }
}

AnitoWave::AnitoWave(const Config &config) : m_config(config), m_width(config.width), m_height(config.height) {
}

AnitoWave::~AnitoWave() {
    delete m_solver;

    if (m_particlePositions) {
        cudaFreeHost(m_particlePositions);
        m_particlePositions = nullptr;
    }

    if (m_particleVelocities)
    {
        cudaFreeHost(m_particleVelocities);
        m_particleVelocities = nullptr;
    }

    if (m_window) {
        bgfx::shutdown();
        glfwTerminate();
    }
}

void AnitoWave::initStaticRenderResources() {
    m_particleLayout.begin()
        .add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
        .add(bgfx::Attrib::Normal, 3, bgfx::AttribType::Float)
        .add(bgfx::Attrib::Color0, 4, bgfx::AttribType::Uint8, true)
        .end();

    m_instanceLayout.begin()
        .add(bgfx::Attrib::TexCoord7, 4, bgfx::AttribType::Float, true)
        .add(bgfx::Attrib::TexCoord6, 4, bgfx::AttribType::Float)
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

    initColliderRendering();

    m_program = loadProgram("vs_fluid_particles", "fs_fluid_particles");

    m_depthProgram = loadProgram("vs_fluid_depth", "fs_fluid_depth");
    m_debugDepthProgram = loadProgram("vs_fullscreen", "fs_debug_depth");

    m_blurProgram = loadProgram("vs_fullscreen", "fs_fluid_blur");
    m_blurParamsUniform = bgfx::createUniform("u_blurParams", bgfx::UniformType::Vec4);
    m_blurFalloffUniform = bgfx::createUniform("u_blurFalloff", bgfx::UniformType::Vec4);

    m_compositeProgram = loadProgram("vs_fullscreen", "fs_fluid_composite");
    m_texelSizeUniform = bgfx::createUniform("u_texelSize", bgfx::UniformType::Vec4);

    m_thicknessProgram = loadProgram("vs_fluid_thickness", "fs_fluid_thickness");
    m_thicknessTexture = bgfx::createTexture2D(
        m_width, m_height, false, 1,
        bgfx::TextureFormat::R16F,
        BGFX_TEXTURE_RT | BGFX_SAMPLER_U_CLAMP | BGFX_SAMPLER_V_CLAMP
    );
    bgfx::TextureHandle thicknessAttachments[] = { m_thicknessTexture, m_hwDepthTexture };
    m_thicknessFrameBuffer = bgfx::createFrameBuffer(BX_COUNTOF(thicknessAttachments), thicknessAttachments, false);
    bgfx::setViewClear(m_kThicknessPassView, BGFX_CLEAR_COLOR, 0x000000FF, 1.0f, 0);
    bgfx::setViewFrameBuffer(m_kThicknessPassView, m_thicknessFrameBuffer);

    m_particleRadiusUniform = bgfx::createUniform("u_particleRadius", bgfx::UniformType::Vec4);
}

void AnitoWave::initColliderRendering() {
    generateCubeTemplate();
}

// void AnitoWave::generateSphereTemplate(int stacks, int slices) {
//     m_circleTemplate.clear();
//     m_circleIndices.clear();
//
//     for (int i = 0; i <= stacks; ++i) {
//         float v = (float)i / (float)stacks;
//         float phi = v * bx::kPi;
//
//         for (int j = 0; j <= slices; ++j) {
//             float u = (float)j / (float)slices;
//             float theta = u * bx::kPi * 2.0f;
//
//             float x = bx::sin(phi) * bx::cos(theta);
//             float y = bx::cos(phi);
//             float z = bx::sin(phi) * bx::sin(theta);
//
//             m_circleTemplate.push_back({x, y, z, x, y, z, 0xffffffff});
//         }
//     }
//
//     for (int i = 0; i < stacks; ++i) {
//         for (int j = 0; j < slices; ++j) {
//             int p1 = i * (slices + 1) + j;
//             int p2 = p1 + (slices + 1);
//
//             m_circleIndices.push_back(p1);
//             m_circleIndices.push_back(p2);
//             m_circleIndices.push_back(p1 + 1);
//
//             m_circleIndices.push_back(p1 + 1);
//             m_circleIndices.push_back(p2);
//             m_circleIndices.push_back(p2 + 1);
//         }
//     }
// }

void AnitoWave::generateSphereTemplate(int stacks, int slices) {
    m_circleTemplate.clear();
    m_circleIndices.clear();

    // A flat 2D square facing the camera
    m_circleTemplate = {
        {-1.0f, -1.0f, 0.0f,  0,0,0, 0xffffffff}, // Bottom-left
        { 1.0f, -1.0f, 0.0f,  0,0,0, 0xffffffff}, // Bottom-right
        {-1.0f,  1.0f, 0.0f,  0,0,0, 0xffffffff}, // Top-left
        { 1.0f,  1.0f, 0.0f,  0,0,0, 0xffffffff}  // Top-right
    };

    m_circleIndices = {0, 1, 2, 1, 3, 2};
}

void AnitoWave::generateCubeTemplate() {
    PosColorVertex vertices[] = {
        // Front face (Normal: 0, 0, 1)
        {-1.0f,  1.0f,  1.0f,  0.0f, 0.0f, 1.0f, 0xffffffff},
        { 1.0f,  1.0f,  1.0f,  0.0f, 0.0f, 1.0f, 0xffffffff},
        {-1.0f, -1.0f,  1.0f,  0.0f, 0.0f, 1.0f, 0xffffffff},
        { 1.0f, -1.0f,  1.0f,  0.0f, 0.0f, 1.0f, 0xffffffff},
        // Back face (Normal: 0, 0, -1)
        { 1.0f,  1.0f, -1.0f,  0.0f, 0.0f, -1.0f, 0xffffffff},
        {-1.0f,  1.0f, -1.0f,  0.0f, 0.0f, -1.0f, 0xffffffff},
        { 1.0f, -1.0f, -1.0f,  0.0f, 0.0f, -1.0f, 0xffffffff},
        {-1.0f, -1.0f, -1.0f,  0.0f, 0.0f, -1.0f, 0xffffffff},
        // Top face (Normal: 0, 1, 0)
        {-1.0f,  1.0f, -1.0f,  0.0f, 1.0f, 0.0f, 0xffffffff},
        { 1.0f,  1.0f, -1.0f,  0.0f, 1.0f, 0.0f, 0xffffffff},
        {-1.0f,  1.0f,  1.0f,  0.0f, 1.0f, 0.0f, 0xffffffff},
        { 1.0f,  1.0f,  1.0f,  0.0f, 1.0f, 0.0f, 0xffffffff},
        // Bottom face (Normal: 0, -1, 0)
        {-1.0f, -1.0f,  1.0f,  0.0f, -1.0f, 0.0f, 0xffffffff},
        { 1.0f, -1.0f,  1.0f,  0.0f, -1.0f, 0.0f, 0xffffffff},
        {-1.0f, -1.0f, -1.0f,  0.0f, -1.0f, 0.0f, 0xffffffff},
        { 1.0f, -1.0f, -1.0f,  0.0f, -1.0f, 0.0f, 0xffffffff},
        // Right face (Normal: 1, 0, 0)
        { 1.0f,  1.0f,  1.0f,  1.0f, 0.0f, 0.0f, 0xffffffff},
        { 1.0f,  1.0f, -1.0f,  1.0f, 0.0f, 0.0f, 0xffffffff},
        { 1.0f, -1.0f,  1.0f,  1.0f, 0.0f, 0.0f, 0xffffffff},
        { 1.0f, -1.0f, -1.0f,  1.0f, 0.0f, 0.0f, 0xffffffff},
        // Left face (Normal: -1, 0, 0)
        {-1.0f,  1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0xffffffff},
        {-1.0f,  1.0f,  1.0f, -1.0f, 0.0f, 0.0f, 0xffffffff},
        {-1.0f, -1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0xffffffff},
        {-1.0f, -1.0f,  1.0f, -1.0f, 0.0f, 0.0f, 0xffffffff},
    };

    const uint32_t indices[] = {
        0,  2,  1,   1,  2,  3,    // Front  (Normal:  0,  0,  1)
        4,  6,  5,   5,  6,  7,    // Back   (Normal:  0,  0, -1)
        8,  10, 9,   9,  10, 11,   // Top    (Normal:  0,  1,  0)
        12, 14, 13,  13, 14, 15,   // Bottom (Normal:  0, -1,  0)
        16, 18, 17,  17, 19, 18,   // Right  (Normal:  1,  0,  0)
        20, 22, 21,  21, 22, 23    // Left   (Normal: -1,  0,  0)
    };

    m_cubeVB = bgfx::createVertexBuffer(
        bgfx::copy(vertices, sizeof(vertices)),
        m_particleLayout
    );
    m_cubeIB = bgfx::createIndexBuffer(
        bgfx::copy(indices, sizeof(indices)),
        BGFX_BUFFER_INDEX32
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
    bgfx::setViewTransform(m_kDepthPassView, view, proj);
    bgfx::setViewTransform(m_kCompositePassView, view, proj);
    bgfx::setViewTransform(m_kThicknessPassView, view, proj);
}

void AnitoWave::drawFullscreenQuad(bgfx::ViewId viewId, bgfx::ProgramHandle program, uint64_t renderState)
{
    if (bgfx::getAvailTransientVertexBuffer(3, m_particleLayout) == 3)
    {
        bgfx::TransientVertexBuffer tvb;
        bgfx::allocTransientVertexBuffer(&tvb, 3, m_particleLayout);
        bgfx::setVertexBuffer(0, &tvb);
        bgfx::setState(renderState);
        bgfx::submit(viewId, program);
    }
}

void AnitoWave::renderFluidComposite()
{
    int kernelSize = (int)m_fluidBlurSize * 2 + 1;
    float sigma = (float)kernelSize / (6.0f * bx::max(0.001f, m_fluidBlurSmoothness));
    // Horizontal Pass
    // Read from Depth, Write to Blur X
    float blurParamsX[4] = { 1.0f / m_width, 0.0f, m_fluidBlurSize, sigma };
    float falloffParams[4] = { m_fluidDepthFactor, 0.0f, 0.0f, 0.0f };
    bgfx::setUniform(m_blurParamsUniform, blurParamsX);
    bgfx::setUniform(m_blurFalloffUniform, falloffParams);
    bgfx::setTexture(0, m_depthSampler, m_depthTexture);
    drawFullscreenQuad(m_kBlurXPassView, m_blurProgram);

    // Vertical Pass
    // Read from Blur X, Write to Blur Y
    float blurParamsY[4] = { 0.0f, 1.0f / m_height, m_fluidBlurSize, sigma };
    bgfx::setUniform(m_blurParamsUniform, blurParamsY);
    bgfx::setUniform(m_blurFalloffUniform, falloffParams);
    bgfx::setTexture(0, m_depthSampler, m_blurXTexture);
    drawFullscreenQuad(m_kBlurYPassView, m_blurProgram);

    // Normals uniform
    float texelData[4] = { 1.0f / m_width, 1.0f / m_height, 0.0f, 0.0f };
    bgfx::setUniform(m_texelSizeUniform, texelData);

    // Composite
    // Read from Blur Y, output to screen
    bgfx::setTexture(0, m_depthSampler, m_blurYTexture);
    bgfx::setTexture(1, m_thicknessSampler, m_thicknessTexture);
    uint64_t compositeState = BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A | BGFX_STATE_BLEND_ALPHA;
    drawFullscreenQuad(m_kCompositePassView, m_compositeProgram, compositeState);
}

void AnitoWave::renderParticles(bgfx::ViewId viewId, bgfx::ProgramHandle program, uint64_t renderState) {
    const size_t numParticles = m_solver->getNumParticles();
    if (numParticles == 0) return;

    std::vector<ParticleInstance> instances;
    instances.reserve(numParticles);

    for (size_t i = 0; i < numParticles; ++i) {
        instances.push_back({
            m_particlePositions[i * 3],
            m_particlePositions[i * 3 + 1],
            m_particlePositions[i * 3 + 2],
            0.0f,
            m_particleVelocities[i * 3 + 0],
            m_particleVelocities[i * 3 + 1],
            m_particleVelocities[i * 3 + 2],
            0.0f
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

            bgfx::setState(renderState);

            bgfx::submit(viewId, program);

            offset += count;
        } else {
            fprintf(stderr, "Instance buffer allocation failed.\n");
        }
    }
}

void AnitoWave::renderFluidPasses() {
    const size_t numParticles = m_solver->getNumParticles();
    if (numParticles == 0) return;

    std::vector<ParticleInstance> instances;
    instances.reserve(numParticles);
    for (size_t i = 0; i < numParticles; ++i) {
        instances.push_back({
            m_particlePositions[i * 3], m_particlePositions[i * 3 + 1], m_particlePositions[i * 3 + 2], 0.0f,
            m_particleVelocities[i * 3 + 0], m_particleVelocities[i * 3 + 1], m_particleVelocities[i * 3 + 2], 0.0f
        });
    }

    uint32_t maxAvailable = bgfx::getAvailInstanceDataBuffer(numParticles, m_instanceLayout.getStride());
    const uint32_t maxInstancesPerBatch = bx::min(maxAvailable, 1048576u);
    uint32_t offset = 0;

    uint64_t depthState = BGFX_STATE_WRITE_R | BGFX_STATE_WRITE_Z | BGFX_STATE_DEPTH_TEST_LESS;
    uint64_t thicknessState = BGFX_STATE_WRITE_R | BGFX_STATE_BLEND_ADD;

    while (offset < numParticles) {
        uint32_t count = bx::min(maxInstancesPerBatch, (uint32_t)numParticles - offset);
        bgfx::InstanceDataBuffer idb{};
        bgfx::allocInstanceDataBuffer(&idb, count, m_instanceLayout.getStride());

        if (idb.data != NULL) {
            memcpy(idb.data, &instances[offset], count * sizeof(ParticleInstance));

            float radiusData[4] = { m_particleRadius, 0.0f, 0.0f, 0.0f };
            bgfx::setUniform(m_particleRadiusUniform, radiusData);

            bgfx::setVertexBuffer(0, m_circleVB);
            bgfx::setIndexBuffer(m_circleIB);
            bgfx::setInstanceDataBuffer(&idb);

            // --- PASS 1: DEPTH ---
            bgfx::setState(depthState);
            bgfx::submit(m_kDepthPassView, m_depthProgram, 0, BGFX_DISCARD_NONE);

            // --- PASS 2: THICKNESS ---
            bgfx::setState(thicknessState);
            bgfx::submit(m_kThicknessPassView, m_thicknessProgram);

            offset += count;
        } else {
            fprintf(stderr, "Instance buffer allocation failed.\n");
            break;
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
        data[0].vx = 1.0f; data[0].vy = 1.0f; data[0].vz = 1.0f; data[0].pad1 = 1.0f;
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
        } else if (col.type == TYPE_MESH) {
            bx::mtxScale(mtxScale, 1.0f, 1.0f, 1.0f);
            bx::mtxMul(mtx, mtxScale, mtxTrans);
            bgfx::setVertexBuffer(0, m_terrainVB);
            bgfx::setIndexBuffer(m_terrainIB);
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
            ImGui::SliderFloat("Viscosity", &params.viscosityStrength, 0.0f, 100.0f);
            ImGui::SliderFloat("Target Density", &params.targetDensity, 5.0f, 2000.0f);
            ImGui::SliderFloat("Pressure Multiplier", &params.pressureMultiplier, 10.0f, 2000.0f);
            ImGui::SliderFloat("Near Pressure Multiplier", &params.nearPressureMultiplier, 0.0f, 100.0f);

            ImGui::Separator();
            ImGui::Text("Collision");
            ImGui::SliderFloat("Collision Damping", &params.collisionDamping, 0.0f, 1.0f);
            ImGui::SliderFloat("Bounds X", &params.boundsX, 1.0f, 80.0f);
            ImGui::SliderFloat("Bounds Y", &params.boundsY, 1.0f, 80.0f);
            ImGui::SliderFloat("Bounds Z", &params.boundsZ, 1.0f, 80.0f);

            ImGui::Separator();
            ImGui::Text("Colliders");
            ImGui::SliderFloat("Collider Drag", &params.colliderDragMultiplier, 0.0f, 0.5f);

            ImGui::Separator();
            if (ImGui::Button("Reset to Defaults")) {
                params = SPHParams();
            }
        }
        ImGui::End();

        ImGui::SetNextWindowPos(ImVec2(10, 430), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(220, 160), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Scenes"))
        {
            for (int i = 0; i < (int)m_scenes.size(); i++)
            {
                bool selected = (i == m_activeScene);
                if (ImGui::Selectable(m_scenes[i].name, selected))
                {
                    if (i != m_activeScene)
                    {
                        m_activeScene = i;
                        bgfx::frame();
                        cudaDeviceSynchronize();
                        loadScene(i);
                    }
                }
            }
        }
        ImGui::End();
    }

    if (m_showRenderEditor)
    {
        ImGui::SetNextWindowPos(ImVec2(320, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(300, 150), ImGuiCond_FirstUseEver);

        if (ImGui::Begin("Rendering Parameters", &m_showRenderEditor)) {
            ImGui::Text("Screen Space Fluid");
            ImGui::SliderFloat("Blur Radius", &m_fluidBlurSize, 1.0f, 100.0f, "%.0f");
            ImGui::SliderFloat("Blur Smoothness", &m_fluidBlurSmoothness, 0.1f, 3.0f);
            ImGui::SliderFloat("Depth Factor (Edge Sharpness)", &m_fluidDepthFactor, 0.0f, 10.0f);
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
    init.limits.maxTransientVbSize = 128 * 1024 * 1024;
    init.limits.maxTransientIbSize = 128 * 1024 * 1024;
    if (!bgfx::init(init)) {
        fprintf(stderr, "Failed to initialize bgfx\n");
        glfwDestroyWindow(m_window);
        glfwTerminate();
        return false;
    }
    // Set view 0 to window dimension
    bgfx::setViewClear(m_kClearView, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, 0x303030ff, 1.0f, 0);
    bgfx::setViewRect(m_kClearView, 0, 0, bgfx::BackbufferRatio::Equal);
    bgfx::setViewRect(m_kThicknessPassView, 0, 0, m_width, m_height);

    bgfx::TextureHandle fbTextures[2];

    fbTextures[0] = bgfx::createTexture2D(
        m_width, m_height, false, 1,
        bgfx::TextureFormat::R32F,
        BGFX_TEXTURE_RT | BGFX_SAMPLER_U_CLAMP | BGFX_SAMPLER_V_CLAMP
    );

    fbTextures[1] = bgfx::createTexture2D(
        m_width, m_height, false, 1,
        bgfx::TextureFormat::D24F,
        BGFX_TEXTURE_RT
    );
    m_depthTexture = fbTextures[0];
    m_hwDepthTexture = fbTextures[1];
    // Bind both to the FBO
    m_depthFbo = bgfx::createFrameBuffer(2, fbTextures, true);

    // Configure the View
    bgfx::setViewClear(m_kDepthPassView, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, 0x00000000, 1.0f, 0);
    bgfx::setViewFrameBuffer(m_kDepthPassView, m_depthFbo);
    bgfx::setViewRect(m_kDepthPassView, 0, 0, m_width, m_height);

    // For debugging depth
    m_depthSampler = bgfx::createUniform("s_depth", bgfx::UniformType::Sampler);
    m_thicknessSampler = bgfx::createUniform("s_thickness", bgfx::UniformType::Sampler);
    bgfx::setViewRect(m_kCompositePassView, 0, 0, bgfx::BackbufferRatio::Equal);

    // --- Blur X Framebuffer ---
    m_blurXTexture = bgfx::createTexture2D(m_width, m_height, false, 1, bgfx::TextureFormat::R32F, BGFX_TEXTURE_RT | BGFX_SAMPLER_U_CLAMP | BGFX_SAMPLER_V_CLAMP);
    m_blurXFbo = bgfx::createFrameBuffer(1, &m_blurXTexture, true);

    // --- Blur Y Framebuffer ---
    m_blurYTexture = bgfx::createTexture2D(m_width, m_height, false, 1, bgfx::TextureFormat::R32F, BGFX_TEXTURE_RT | BGFX_SAMPLER_U_CLAMP | BGFX_SAMPLER_V_CLAMP);
    m_blurYFbo = bgfx::createFrameBuffer(1, &m_blurYTexture, true);

    bgfx::setViewClear(m_kBlurXPassView, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, 0x00000000, 1.0f, 0);
    bgfx::setViewFrameBuffer(m_kBlurXPassView, m_blurXFbo);
    bgfx::setViewRect(m_kBlurXPassView, 0, 0, m_width, m_height);

    bgfx::setViewClear(m_kBlurYPassView, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, 0x00000000, 1.0f, 0);
    bgfx::setViewFrameBuffer(m_kBlurYPassView, m_blurYFbo);
    bgfx::setViewRect(m_kBlurYPassView, 0, 0, m_width, m_height);

    imguiCreate();

    initStaticRenderResources();

    buildSceneList();
    loadScene(0);

    updateCamera();

    return true;
}

void AnitoWave::run() {
    double lastTime = glfwGetTime();
    double accumulator = 0.0f;

    const float FIXED_DT = 1.0f / 180.0f;
    const int MAX_STEPS_PER_FRAME = 15;

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

        if (frameTime > 0.1) frameTime = 0.1;

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

        // if (accumulator > FIXED_DT) {
        //     accumulator = 0.0f;
        // }

        m_solver->getPositions(m_particlePositions);
        m_solver->getVelocities(m_particleVelocities);

        cudaStreamSynchronize(0);

        // uint64_t depthState = BGFX_STATE_WRITE_R | BGFX_STATE_WRITE_Z | BGFX_STATE_DEPTH_TEST_LESS;
        // renderParticles(m_kDepthPassView, m_depthProgram, depthState);
        //
        // uint64_t thicknessState = BGFX_STATE_WRITE_R | BGFX_STATE_BLEND_ADD;
        // renderParticles(m_kThicknessPassView, m_thicknessProgram, thicknessState);
        renderFluidPasses();

        renderColliders();
        renderFluidComposite();

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

    if (app && key == GLFW_KEY_F3 && action == GLFW_RELEASE) {
        app->m_showRenderEditor = !app->m_showRenderEditor;
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