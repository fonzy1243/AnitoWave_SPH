#include <sph.cuh>
#include <bx/math.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <cmath>

__device__ float2 operator+(float2 a, float2 b) {
    return float2(a.x + b.x, a.y + b.y);
}
__device__ float2 operator-(float2 a, float2 b) {
    return float2(a.x - b.x, a.y - b.y);
}

__device__ float2 operator*(float2 a, float s) {
    return float2(a.x * s, a.y * s);
}

__device__ float2 operator*(float s, float2 a) {
    return float2(a.x * s, a.y * s);
}

__device__ float dot(float2 a, float2 b) {
    return a.x * b.x + a.y * b.y;
}

__device__ float length(float2 v) {
    return sqrtf(v.x * v.x + v.y * v.y);
}

__device__ float sign(float x) {
    return x < 0.0f ? -1.0f : 1.0f;
}

__device__ void ResolveCollisions(float* positions, float* velocities, int numParticles, float particleSize, float boundsX, float boundsY, float collisionDamping) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    int idx = i * 2;

    float2 halfBoundsSize = make_float2(boundsX / 2 - 1 * particleSize, boundsY / 2 - 1 * particleSize);

    if (abs(positions[idx]) > halfBoundsSize.x) {
        positions[idx] = halfBoundsSize.x * sign(positions[idx]);
        velocities[idx] *= -1 * collisionDamping;
    }
    if (abs(positions[idx + 1]) > halfBoundsSize.y) {
        positions[idx + 1] = halfBoundsSize.y * sign(positions[idx + 1]);
        velocities[idx + 1] *= -1 * collisionDamping;
    }
}

__global__ void UpdatePositions(float* positions, float* velocities, int numParticles, float particleSize, float boundsX, float boundsY,
    float collisionDamping, float gravity, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    int idx = i * 2;

    velocities[idx + 1] += -gravity * dt;
    positions[idx] += velocities[idx] * dt;
    positions[idx + 1] += velocities[idx + 1] * dt;

    ResolveCollisions(positions, velocities, numParticles, particleSize, boundsX, boundsY, collisionDamping);
}

SPHSolver::SPHSolver(int maxParticles) : m_maxParticles(maxParticles), m_numParticles(0) {
    m_params.particleSize = 0.02f;

    size_t size = m_maxParticles * 2 * sizeof(float);
    cudaMalloc(&d_positions, size);
    cudaMalloc(&d_velocities, size);
}

SPHSolver::~SPHSolver() {
    if (d_positions) cudaFree(d_positions);
    if (d_velocities) cudaFree(d_velocities);
}

void SPHSolver::init(const std::vector<float> &positions, const std::vector<float> &velocities) {
    m_numParticles = positions.size() / 2;
    if (m_numParticles > m_maxParticles) m_numParticles = m_maxParticles;

    size_t copySize = m_numParticles * 2 * sizeof(float);
    cudaMemcpy(d_positions, positions.data(), copySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocities, velocities.data(), copySize, cudaMemcpyHostToDevice);
}

void SPHSolver::update(float dt) {
    int blockSize = 256;
    int numBlock = (m_numParticles + blockSize - 1) / blockSize;

    // Update positions and handle collisions
    UpdatePositions<<<numBlock, blockSize>>>(d_positions, d_velocities, m_numParticles, m_params.particleSize, m_params.boundsX, m_params.boundsY, m_params.collisionDamping, m_params.gravity, m_params.dt);
    cudaDeviceSynchronize();
}

void SPHSolver::getPositions(std::vector<float> &outPositions) {
    if (outPositions.size() != m_numParticles * 2) {
        outPositions.resize(m_numParticles * 2);
    }
    // TODO: Find a way to eliminate the transfer to CPU memory for rendering
    cudaMemcpy(outPositions.data(), d_positions, m_numParticles * 2 * sizeof(float), cudaMemcpyDeviceToHost);
}
