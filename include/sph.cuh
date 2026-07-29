#ifndef ANITOWAVE_SPH_SPH_CUH
#define ANITOWAVE_SPH_SPH_CUH

#include <tiny_obj_loader.h>
#include <tiny_gltf.h>
#include <tmd/TriangleMeshDistance.h>
#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <algorithm>
#include <numbers>
#include <bx/math.h>
#include <cub/cub.cuh>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "bgfx/bgfx.h"

struct PosColorVertex {
    float x, y, z;
    float nx, ny, nz;
    uint32_t abgr;
    float u, v;
};

struct MeshDrawGroup
{
    uint32_t indexStart;
    uint32_t indexCount;
    int textureIndex;
};

struct MeshSDFData {
    std::vector<float> distanceData;
    int resX, resY, resZ;
    float3 minBounds;
    float3 maxBounds;

    // rendering data
    std::vector<PosColorVertex> renderVertices;
    std::vector<uint32_t> renderIndices;

    // Per-primitive draw groups
    std::vector<MeshDrawGroup> drawGroups;

    // Per-image texture data
    std::vector<std::vector<uint8_t>> texturePixels;
    std::vector<int> textureWidths;
    std::vector<int> textureHeights;
};

enum ColliderType { TYPE_SPHERE = 0, TYPE_BOX = 1, TYPE_MESH = 2 };
struct Collider {
    ColliderType type;
    float3 position;
    float3 dims;

    bool isDynamic;
    float mass;
    float3 velocity;
    float3 forceAccumulator;

    cudaTextureObject_t sdfTexture;
    cudaArray_t sdfArray;
    float3 gridMinBounds;
    float3 gridMaxBounds;
    float3 voxelSize;
};

struct SPHParams {
    float particleSize = 0.05f;
    float gravity = 0.0f;
    float collisionDamping = 0.95f;
    float predictFactor = 1/120.0f;
    float boundsX = 10.0f;
    float boundsY = 10.0f;
    float boundsZ = 10.0f;
    float smoothingRadius = 0.25;
    float targetDensity = 255.5f;
    float pressureMultiplier = 650.0f;
    float viscosityStrength = 02.0f;
    float nearPressureMultiplier = 0.5f;
    float colliderDragMultiplier = 0.001f;

    float densityScale;        // For SmoothingKernel
    float pressureScale;       // For SmoothingKernelDerivative
    float viscosityScale;      // For ViscositySmoothingKernel
    float nearDensityScale;    // For NearDensityKernel
    float nearPressureScale;   // For NearDensityDerivativeKernel
};

struct WhiteParticle
{
    float3 position;
    float3 velocity;
    float remainingLifetime;
    float pad;
};

struct WhiteParticleParams
{
    float trappedAirMin = 5.0f;
    float trappedAirMax = 20.0f;
    float trappedAirSpawnRate = 50.0f;
    float kineticEnergyMin = 2.0f;
    float kineticEnergyMax = 20.0f;
    int bubbleThreshold = 20;
    int sprayThreshold = 6;
    uint32_t maxWhiteParticles = 131072;
};

class SPHSolver {
public:
    SPHSolver(int maxParticles);
    ~SPHSolver();

    void init(const std::vector<float>& positions, const std::vector<float>& velocities);
    void update(float dt);
    void UpdateSpatialLookup();
    void addCollider(Collider collider);

    void updateWhiteParticles(float dt);
    void getWhiteParticles(std::vector<WhiteParticle>& out);
    uint32_t getWhiteParticleCount();

    void setParams(const SPHParams& params);
    SPHParams& getParams() { return m_params; }
    void getColliders(std::vector<Collider>& outColliders);

    void getPositions(float* outPositions);
    void getVelocities(float* outVelocities);

    int getNumParticles() const { return m_numParticles; }

    void getWhiteParticles(float* outPositions, uint32_t& outCount);

    void requestWhiteParticles();
    void finalizeWhiteParticles(std::vector<WhiteParticle>& out);

private:
    int m_numParticles;
    int m_maxParticles;
    int m_numColliders = 0;
    uint32_t m_hashTableSize;
    SPHParams m_params;

    // White particles (foam/spray/bubbles)
    WhiteParticle* d_whiteParticles = nullptr;
    WhiteParticle* d_whiteParticlesCompact = nullptr;
    WhiteParticle* h_whiteParticlesPinned = nullptr;
    uint32_t* h_whiteCountPinned = nullptr;
    uint32_t* d_whiteCounters = nullptr;
    uint32_t m_maxWhiteParticles = 65536;
    WhiteParticleParams m_wpParams;
    float m_simTime = 0.0f;
    float* d_trappedAir = nullptr;

    // Solids
    std::vector<Collider> m_colliders;
    Collider* d_colliders = nullptr;

    // Physical values
    float *d_posX, *d_posY, *d_posZ;
    float *d_sortedPosX, *d_sortedPosY, *d_sortedPosZ;
    float *d_predX, *d_predY, *d_predZ;
    float *d_velX, *d_velY, *d_velZ;
    float* d_densities;
    float* d_nearDensities;
    // Spatial hashing
    uint32_t* d_spatialIndices;
    uint32_t* d_spatialKeys;
    uint32_t* d_spatialIndicesSorted;
    uint32_t* d_spatialKeysSorted;
    uint32_t* d_startIndices;
    // Sorted buffers
    float *d_sortedPredX, *d_sortedPredY, *d_sortedPredZ;
    float *d_sortedVelX, *d_sortedVelY, *d_sortedVelZ;
    // AOS buffer for rendering
    float* d_aos_temp;

    // For sorting
    void* d_sortStorage = nullptr;
    size_t m_sortStorageBytes = 0;
};

#endif //ANITOWAVE_SPH_SPH_CUH