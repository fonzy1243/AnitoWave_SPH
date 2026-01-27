#include <sph.cuh>
#include <numbers>
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

__device__ float2 operator/(float2 a, float s) {
    return float2(a.x / s, a.y / s);
}

__device__ float2 operator+=(float2& a, float2 b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}

__device__ float2 operator+=(float2& a, float b) {
    a.x += b;
    a.y += b;
    return a;
}

__device__ float dot(float2 a, float2 b) {
    return a.x * b.x + a.y * b.y;
}

__device__ float length(float2 v) {
    return sqrtf(v.x * v.x + v.y * v.y);
}

__device__ float lengthSqr(float2 v) {
    return v.x * v.x + v.y * v.y;
}

__device__ float sign(float x) {
    return x < 0.0f ? -1.0f : 1.0f;
}

__device__ int2 PositionToCellCoord(float2 point, float radius) {
    int cellX = (int)(point.x / radius);
    int cellY = (int)(point.y / radius);
    return make_int2(cellX, cellY);
}

__device__ uint32_t HashCell(int cellX, int cellY) {
    uint32_t a = (uint32_t)cellX * 15823;
    uint32_t b = (uint32_t)cellY * 9737333;
    return a + b;
}

__device__ uint32_t GetKeyFromHash(uint32_t hash, int numParticles) {
    return hash % (uint32_t)numParticles;
}

__device__ float2 ConvertDensityToPressure(float density, float nearDensity, float targetDensity, float pressureMultiplier, float nearPressureMultiplier) {
    float densityError = density  - targetDensity;
    float pressure = densityError  * pressureMultiplier;
    float nearPressure = nearDensity * nearPressureMultiplier;

    return make_float2(pressure, nearPressure);
}

__device__ float2 CalculateSharedPressure(float densityA, float nearDensityA, float densityB, float nearDensityB, float targetDensity, float pressureMultiplier, float nearPressureMultiplier) {
    float2 pressureA = ConvertDensityToPressure(densityA, nearDensityA, targetDensity, pressureMultiplier, nearPressureMultiplier);
    float2 pressureB = ConvertDensityToPressure(densityB, nearDensityB, targetDensity, pressureMultiplier, nearPressureMultiplier);

    return make_float2(
        (pressureA.x + pressureB.x) / 2,
        (pressureA.y + pressureB.y) / 2
    );
}

__device__ float SmoothingKernel(float dst, float radius) {
    if (dst >= radius) return 0.0f;

    float v = radius - dst;
    return v * v * 6 / (std::numbers::pi_v<float> * powf(radius, 4.0f));
}

__device__ float SmoothingKernelDerivative(float dst, float radius) {
    if (dst >= radius) return 0.0f;

    float v = radius - dst;
    return -v * 12 / (std::numbers::pi_v<float> * powf(radius, 4.0f));
}

__device__ float ViscositySmoothingKernel(float dst, float radius) {
    if (dst >= radius) return 0.0f;

    float volume = (std::numbers::pi_v<float> * powf(radius, 4)) / 6.0f;
    return (radius - dst) * (radius - dst) / volume;
}

__device__ float NearDensityKernel(float dst, float radius) {
    if (dst >= radius) return 0.0f;

    float v = radius - dst;
    return v * v * v * 10 / (std::numbers::pi_v<float> * powf(radius, 5.0f));
}

__device__ float NearDensityDerivativeKernel(float dst, float radius) {
    if (dst >= radius) return 0.0f;

    float v = radius - dst;
    return -v * v * 30 / (std::numbers::pi_v<float> * powf(radius, 5.0f));
}

__device__ float2 CalculateDensity(float* positions, uint32_t* spatialIndices, uint32_t* spatialKeys, uint32_t* startIndices, int numParticles, float2 samplePoint, float smoothingRadius) {
    float density = 0.0f;
    float nearDensity = 0.0f;
    const float mass = 1.0f;

    int2 center = PositionToCellCoord(samplePoint, smoothingRadius);
    float sqrRadius = smoothingRadius * smoothingRadius;

    for (int offsetX = -1; offsetX <= 1; offsetX++) {
        for (int offsetY = -1; offsetY <= 1; offsetY++) {
            int2 cellCoord = make_int2(center.x + offsetX, center.y + offsetY);
            uint32_t hash = HashCell(cellCoord.x, cellCoord.y);
            uint32_t key = GetKeyFromHash(hash, numParticles);

            uint32_t startIndex = startIndices[key];

            for (uint32_t i = startIndex; i < numParticles; i++) {
                if (spatialKeys[i] != key) break;

                uint32_t particleIndex = spatialIndices[i];

                uint32_t idx = particleIndex * 2;
                float2 particlePos = make_float2(positions[idx], positions[idx + 1]);

                float sqrDst = lengthSqr(particlePos - samplePoint);

                if (sqrDst <= sqrRadius) {
                    float dst = sqrtf(sqrDst);
                    float influence = SmoothingKernel(dst, smoothingRadius);
                    density += mass * influence;
                    nearDensity += mass * NearDensityKernel(dst, smoothingRadius);
                }
            }
        }

    }

    return make_float2(density, nearDensity);
}

__device__ float2 CalculatePressureForce(int particleIndex, float* positions, float* densities, float* nearDensities, uint32_t* spatialIndices, uint32_t* spatialKeys, uint32_t* startIndices, int numParticles, float smoothingRadius, float targetDensity, float pressureMultiplier, float nearPressureMultiplier) {
    float2 pressureForce = make_float2(0, 0);
    const float mass = 1.0f;

    int currentIdx = particleIndex * 2;
    float2 samplePoint = make_float2(positions[currentIdx], positions[currentIdx + 1]);
    float density = densities[particleIndex];
    float nearDensity = nearDensities[particleIndex];

    int2 center = PositionToCellCoord(samplePoint, smoothingRadius);
    float sqrRadius = smoothingRadius * smoothingRadius;

    for (int offsetX = -1; offsetX <= 1; offsetX++) {
        for (int offsetY = -1; offsetY <= 1; offsetY++) {
            int2 cellCoord = make_int2(center.x + offsetX, center.y + offsetY);
            uint32_t key = GetKeyFromHash(HashCell(cellCoord.x, cellCoord.y), numParticles);
            uint32_t startIndex = startIndices[key];

            for (uint32_t i = startIndex; i < numParticles; i++) {
                if (spatialKeys[i] != key) break;

                uint32_t otherParticleIndex = spatialIndices[i];
                if (particleIndex == otherParticleIndex) continue;

                uint32_t otherIdx = otherParticleIndex * 2;
                float2 otherPos = make_float2(positions[otherIdx], positions[otherIdx + 1]);
                float2 offset = otherPos - samplePoint;
                float sqrDst = lengthSqr(offset);

                if (sqrDst <= sqrRadius) {
                    float dst = sqrtf(sqrDst);
                    float2 dir = dst == 0.0f ? make_float2(1.0f, 0.0f) : offset / dst;
                    float slope = SmoothingKernelDerivative(dst, smoothingRadius);
                    float nearSlope = NearDensityDerivativeKernel(dst, smoothingRadius);
                    float otherDensity = densities[otherParticleIndex];
                    float otherNearDensity = nearDensities[otherParticleIndex];
                    float2 sharedPressure = CalculateSharedPressure(density, nearDensity, otherDensity, otherNearDensity, targetDensity, pressureMultiplier, nearPressureMultiplier);

                    float combinedForce = (sharedPressure.x * slope) + (sharedPressure.y * nearSlope);

                    pressureForce += combinedForce * dir * mass / otherDensity;
                }
            }
        }
    }

    return pressureForce;
}

__device__ float2 CalculateViscosityForce(int particleIndex, float* positions, float* velocities,
    uint32_t* spatialIndices, uint32_t* spatialKeys, uint32_t* startIndices,
    int numParticles, float smoothingRadius, float viscosityStrength) {
    float2 viscosityForce = make_float2(0, 0);
    int currentIdx = particleIndex * 2;

    float2 position = make_float2(positions[currentIdx], positions[currentIdx + 1]);
    float2 velocity = make_float2(velocities[currentIdx], velocities[currentIdx + 1]);

    int2 center = PositionToCellCoord(position, smoothingRadius);
    float sqrRadius = smoothingRadius * smoothingRadius;

    for (int offsetX = -1; offsetX <= 1; offsetX++) {
        for (int offsetY = -1; offsetY <= 1; offsetY++) {
            int2 cellCoord = make_int2(center.x + offsetX, center.y + offsetY);
            uint32_t key = GetKeyFromHash(HashCell(cellCoord.x, cellCoord.y), numParticles);
            uint32_t startIndex = startIndices[key];

            for (uint32_t i = startIndex; i < numParticles; i++) {
                if (spatialKeys[i] != key) break;

                uint32_t otherParticleIndex = spatialIndices[i];
                if (particleIndex == otherParticleIndex) continue;

                int otherIdx = otherParticleIndex * 2;
                float2 otherPos = make_float2(positions[otherIdx], positions[otherIdx + 1]);
                float2 otherVel = make_float2(velocities[otherIdx], velocities[otherIdx + 1]);

                float sqrDst = lengthSqr(otherPos - position);

                if (sqrDst <= sqrRadius) {
                    float dst = sqrtf(sqrDst);
                    float influence = SmoothingKernel(dst, smoothingRadius);

                    viscosityForce += (otherVel - velocity) * influence;
                }
            }
        }
    }

    return viscosityForce * viscosityStrength;
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

__global__ void ApplyPressureForces(float* positions, float* velocities, float* densities, float* nearDensities, uint32_t* spatialIndices,
    uint32_t* spatialKeys, uint32_t* startIndices, int numParticles, float smoothingRadius,
    float targetDensity, float pressureMultiplier, float nearPressureMultiplier, float viscosityStrength, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    float2 pressureForce = CalculatePressureForce(i, positions, densities, nearDensities, spatialIndices, spatialKeys, startIndices, numParticles, smoothingRadius, targetDensity, pressureMultiplier, nearPressureMultiplier);
    float2 viscosityForce = CalculateViscosityForce(i, positions, velocities, spatialIndices, spatialKeys, startIndices, numParticles, smoothingRadius, viscosityStrength);

    float2 totalForce = pressureForce + viscosityForce;
    float2 acceleration = totalForce / densities[i];

    int idx = i * 2;
    velocities[idx] += acceleration.x * dt;
    velocities[idx + 1] += acceleration.y * dt;
}

__global__ void UpdatePositions(float* positions, float* velocities, int numParticles, float particleSize, float boundsX, float boundsY,
    float collisionDamping, float gravity, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    int idx = i * 2;

    positions[idx] += velocities[idx] * dt;
    positions[idx + 1] += velocities[idx + 1] * dt;
    ResolveCollisions(positions, velocities, numParticles, particleSize, boundsX, boundsY, collisionDamping);
}

__global__ void PredictPositions(float* positions, float* predictedPositions, float* velocities, int numParticles, float gravity, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    int idx = i * 2;

    velocities[idx + 1] += -1 * gravity * dt;
    predictedPositions[idx] = positions[idx] + velocities[idx] * 1 / 120.0f;
    predictedPositions[idx + 1] = positions[idx + 1] + velocities[idx + 1] * 1 / 120.0f;
}

__global__ void UpdateDensities(float* positions, float* velocities, float* densities, float* nearDensities, uint32_t* spatialIndices, uint32_t* spatialKeys, uint32_t* startIndices, int numParticles, float smoothingRadius, float gravity, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    int idx = i * 2;

    float2 samplePoint  = make_float2(positions[idx], positions[idx + 1]);
    float2 result = CalculateDensity(positions, spatialIndices, spatialKeys, startIndices, numParticles, samplePoint, smoothingRadius);
    densities[i] = result.x;
    nearDensities[i] = result.y;
}

__global__ void UpdateSpatialHash(float* positions, int numParticles, float radius, uint32_t* spatialIndices, uint32_t* spatialKeys, uint32_t* startIndices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    int idx = i * 2;
    float2 pos = make_float2(positions[idx], positions[idx + 1]);

    startIndices[i] = 0xffffffff;

    int2 cell = PositionToCellCoord(pos, radius);
    uint32_t cellKey = GetKeyFromHash(HashCell(cell.x, cell.y), numParticles);

    spatialIndices[i] = i;
    spatialKeys[i] = cellKey;
}

__global__ void UpdateStartIndices(uint32_t* spatialKeys, uint32_t* startIndices, int numParticles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    uint32_t key = spatialKeys[i];
    uint32_t keyPrev = i == 0 ? 0xffffffff : spatialKeys[i - 1];
    if (key != keyPrev) {
        startIndices[key] = i;
    }
}

// Pseudocode for using the spatial hashing
// __device__ ForeachPointWithinRadius(float2 samplePoint) {
//     int2 center = PositionToCellCoord(samplePoint, radius);
//     float sqrRadius = radius * radius;
//
//     foreach ((int offsetX, int offsetY) in cell) {
//         uint32_t key = GetKeyFromHash(HashCell(center.x + offsetX, center.y + offsetY), numParticles);
//         int cellStartIndex = startIndices[key];
//
//         for (int i = cellStartIndex; i < numParticles; i++) {
//             if (spatialLookup[i].cellKey != key) break;
//
//             int particleIndex = spatialLookup[i].particleIndex;
//             float sqrDst = (points[particleIndex] - samplePoint).sqrMagnitude;
//
//             // Test if point is inside radius
//             if (sqrtDst <= sqrRadius) {
//                 // Do something with particle index
//             }
//         }
//     }
// }

void SPHSolver::UpdateSpatialLookup() {
    int blockSize = 256;
    int numBlock = (m_numParticles + blockSize - 1) / blockSize;

    UpdateSpatialHash<<<numBlock, blockSize>>>(d_predictedPositions, m_numParticles, m_params.smoothingRadius, d_spatialIndices, d_spatialKeys, d_startIndices);
    cudaDeviceSynchronize();

    // Sort by cell key
    thrust::device_ptr<uint32_t> t_keys(d_spatialKeys);
    thrust::device_ptr<uint32_t> t_indices(d_spatialIndices);

    thrust::sort_by_key(t_keys, t_keys + m_numParticles, t_indices);

    UpdateStartIndices<<<numBlock, blockSize>>>(d_spatialKeys, d_startIndices, m_numParticles);
    cudaDeviceSynchronize();
}

SPHSolver::SPHSolver(int maxParticles) : m_maxParticles(maxParticles), m_numParticles(0) {
    size_t size = m_maxParticles * 2 * sizeof(float);
    cudaMalloc(&d_positions, size);
    cudaMalloc(&d_predictedPositions, size);
    cudaMalloc(&d_velocities, size);
    cudaMalloc(&d_densities, m_maxParticles * sizeof(float));
    cudaMalloc(&d_nearDensities, m_maxParticles * sizeof(float));
    cudaMalloc(&d_spatialIndices, m_maxParticles * sizeof(uint32_t));
    cudaMalloc(&d_spatialKeys, m_maxParticles * sizeof(uint32_t));
    cudaMalloc(&d_startIndices, m_maxParticles * sizeof(uint32_t));
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

    // Apply gravity and predict next positions
    PredictPositions<<<numBlock, blockSize>>>(d_positions, d_predictedPositions, d_velocities,
        m_numParticles, m_params.gravity, dt);
    cudaDeviceSynchronize();

    UpdateSpatialLookup();

    // Calculate and apply densities
    UpdateDensities<<<numBlock, blockSize>>>(d_predictedPositions, d_velocities, d_densities, d_nearDensities,
        d_spatialIndices, d_spatialKeys, d_startIndices, m_numParticles,
        m_params.smoothingRadius, m_params.gravity, dt);
    cudaDeviceSynchronize();

    // Calculate and apply pressure forces
    ApplyPressureForces<<<numBlock, blockSize>>>(d_predictedPositions, d_velocities, d_densities, d_nearDensities,
        d_spatialIndices, d_spatialKeys, d_startIndices, m_numParticles,
        m_params.smoothingRadius, m_params.targetDensity, m_params.pressureMultiplier, m_params.nearPressureMultiplier, m_params.viscosityStrength, dt);
    cudaDeviceSynchronize();

    // Update positions and handle collisions
    UpdatePositions<<<numBlock, blockSize>>>(d_positions, d_velocities, m_numParticles,
        m_params.particleSize, m_params.boundsX, m_params.boundsY,
        m_params.collisionDamping, m_params.gravity, dt);
    cudaDeviceSynchronize();
}

void SPHSolver::getPositions(std::vector<float> &outPositions) {
    if (outPositions.size() != m_numParticles * 2) {
        outPositions.resize(m_numParticles * 2);
    }
    // TODO: Find a way to eliminate the transfer to CPU memory for rendering
    cudaMemcpy(outPositions.data(), d_positions, m_numParticles * 2 * sizeof(float), cudaMemcpyDeviceToHost);
}

void SPHSolver::setParams(const SPHParams &params) {
    m_params = params;
}
