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
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

struct PosColorVertex {
    float x, y, z;
    float nx, ny, nz;
    uint32_t abgr;
};

struct MeshSDFData {
    std::vector<float> distanceData;
    int resX, resY, resZ;
    float3 minBounds;
    float3 maxBounds;

    // rendering data
    std::vector<PosColorVertex> renderVertices;
    std::vector<uint32_t> renderIndices;
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
    float particleSize = 0.09f;
    float gravity = 0.0f;
    float collisionDamping = 0.15f;
    float predictFactor = 1/120.0f;
    float boundsX = 10.0f;
    float boundsY = 10.0f;
    float boundsZ = 10.0f;
    float smoothingRadius = 0.15;
    float targetDensity = 855.5f;
    float pressureMultiplier = 150.0f;
    float viscosityStrength = 0.1f;
    float nearPressureMultiplier = 0.1f;
    float colliderDragMultiplier = 0.001f;

    float densityScale;        // For SmoothingKernel
    float pressureScale;       // For SmoothingKernelDerivative
    float viscosityScale;      // For ViscositySmoothingKernel
    float nearDensityScale;    // For NearDensityKernel
    float nearPressureScale;   // For NearDensityDerivativeKernel
};

class SPHSolver {
public:
    SPHSolver(int maxParticles);
    ~SPHSolver();

    void init(const std::vector<float>& positions, const std::vector<float>& velocities);
    void update(float dt);
    void UpdateSpatialLookup();
    void addCollider(Collider collider);

    void setParams(const SPHParams& params);
    SPHParams& getParams() { return m_params; }
    void getColliders(std::vector<Collider>& outColliders);

    void getPositions(float* outPositions);

    int getNumParticles() const { return m_numParticles; }

private:
    int m_numParticles;
    int m_maxParticles;
    int m_numColliders = 0;
    uint32_t m_hashTableSize;
    SPHParams m_params;

    // Solids
    std::vector<Collider> m_colliders;
    Collider* d_colliders = nullptr;

    // Physical values
    float *d_posX, *d_posY, *d_posZ;
    float *d_predX, *d_predY, *d_predZ;
    float *d_velX, *d_velY, *d_velZ;
    float* d_densities;
    float* d_nearDensities;
    // Spatial hashing
    uint32_t* d_spatialIndices;
    uint32_t* d_spatialKeys;
    uint32_t* d_startIndices;
    // Sorted buffers
    float *d_sortedPredX, *d_sortedPredY, *d_sortedPredZ;
    float *d_sortedVelX, *d_sortedVelY, *d_sortedVelZ;
    // AOS buffer for rendering
    float* d_aos_temp;
};

#endif //ANITOWAVE_SPH_SPH_CUH