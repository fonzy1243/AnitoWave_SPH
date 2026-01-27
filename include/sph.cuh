#ifndef ANITOWAVE_SPH_SPH_CUH
#define ANITOWAVE_SPH_SPH_CUH

#include <vector>

struct SPHParams {
    float particleSize = 0.09f;
    float gravity = 0.0f;
    float collisionDamping = 0.95f;
    float predictFactor = 1/120.0f;
    float boundsX = 3.5f;
    float boundsY = 2.0f;
    float smoothingRadius = 0.35;
    float targetDensity = 855.5f;
    float pressureMultiplier = 150.0f;
    float viscosityStrength = 0.1f;
    float nearPressureMultiplier = 0.1f;
};

class SPHSolver {
public:
    SPHSolver(int maxParticles);
    ~SPHSolver();

    void init(const std::vector<float>& positions, const std::vector<float>& velocities);
    void update(float dt);
    void UpdateSpatialLookup();

    void setParams(const SPHParams& params);
    SPHParams& getParams() { return m_params; }

    void getPositions(std::vector<float>& outPositions);

private:
    int m_numParticles;
    int m_maxParticles;
    SPHParams m_params;

    // Device memory
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
};

#endif //ANITOWAVE_SPH_SPH_CUH