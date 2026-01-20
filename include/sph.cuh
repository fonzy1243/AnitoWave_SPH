#ifndef ANITOWAVE_SPH_SPH_CUH
#define ANITOWAVE_SPH_SPH_CUH

#include <vector>

struct SPHParams {
    float particleSize = 1.0f;
    float gravity = 9.81f;
    float collisionDamping = 0.95f;
    float dt = 0.016f;
    float boundsX = 1.6f;
    float boundsY = 0.9f;
};

class SPHSolver {
public:
    SPHSolver(int maxParticles);
    ~SPHSolver();

    void init(const std::vector<float>& positions, const std::vector<float>& velocities);
    void update(float dt);

    void setParams(const SPHParams& params);
    SPHParams& getParams() { return m_params; }

    void getPositions(std::vector<float>& outPositions);

private:
    int m_numParticles;
    int m_maxParticles;
    float m_particleSize;
    SPHParams m_params;

    // Device memory
    float* d_positions;
    float* d_velocities;
};

#endif //ANITOWAVE_SPH_SPH_CUH