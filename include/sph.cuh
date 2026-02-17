#ifndef ANITOWAVE_SPH_SPH_CUH
#define ANITOWAVE_SPH_SPH_CUH

#include <vector>

enum ColliderType { TYPE_SPHERE = 0, TYPE_BOX = 1 };
struct Collider {
    ColliderType type;
    float3 position;
    float3 dims;

    bool isDynamic;
    float mass;
    float3 velocity;
    float3 forceAccumulator;
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
    float* d_positions;
    float* d_predictedPositions;
    float* d_velocities;
    float* d_densities;
    float* d_nearDensities;
    // Spatial hashing
    uint32_t* d_spatialIndices;
    uint32_t* d_spatialKeys;
    uint32_t* d_startIndices;
    // Sorted buffers
    float* d_sortedPredictedPositions;
    float* d_sortedVelocities;
};

#endif //ANITOWAVE_SPH_SPH_CUH