#include <sph.cuh>
#include <numbers>
#include <bx/math.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

__device__ const int3 CELL_OFFSETS[27] = {
    {-1,-1,-1}, { 0,-1,-1}, { 1,-1,-1},
    {-1, 0,-1}, { 0, 0,-1}, { 1, 0,-1},
    {-1, 1,-1}, { 0, 1,-1}, { 1, 1,-1},
    {-1,-1, 0}, { 0,-1, 0}, { 1,-1, 0},
    {-1, 0, 0}, { 0, 0, 0}, { 1, 0, 0},
    {-1, 1, 0}, { 0, 1, 0}, { 1, 1, 0},
    {-1,-1, 1}, { 0,-1, 1}, { 1,-1, 1},
    {-1, 0, 1}, { 0, 0, 1}, { 1, 0, 1},
    {-1, 1, 1}, { 0, 1, 1}, { 1, 1, 1}
};

__device__ inline float3 loadFloat3(float* data, int idx, int limit) {
    if (idx < limit) {
        float x = __ldg(&data[idx * 3 + 0]);
        float y = __ldg(&data[idx * 3 + 1]);
        float z = __ldg(&data[idx * 3 + 2]);
        return make_float3(x, y, z);
    }
    return make_float3(0, 0, 0);
}

__device__ inline float loadFloat(float* data, int idx, int limit) {
    if (idx < limit) return __ldg(&data[idx]);
    return 0.0f;
}

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

__device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator*(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ float3 operator*(float s, float3 a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ float3 operator/(float3 a, float s) {
    return make_float3(a.x / s, a.y / s, a.z / s);
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

__device__ float3& operator+=(float3& a, float3 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__device__ float3& operator+=(float3& a, float b) {
    a.x += b;
    a.y += b;
    a.z += b;
    return a;
}

__device__ float3& operator-=(float3& a, float3 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

__device__ float3& operator-=(float3& a, float b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
    return a;
}

__device__ float3& operator*=(float3& a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
}

__device__ float3 operator-(float3 a) {
    return make_float3(-a.x, -a.y, -a.z);
}

__device__ int3 operator+(int3 a, int3 b) {
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float dot(float2 a, float2 b) {
    return a.x * b.x + a.y * b.y;
}

__device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float length(float2 v) {
    return sqrtf(v.x * v.x + v.y * v.y);
}

__device__ float length(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ float lengthSqr(float2 v) {
    return v.x * v.x + v.y * v.y;
}

__device__ float lengthSqr(float3 v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

__device__ float sign(float x) {
    return x < 0.0f ? -1.0f : 1.0f;
}

__device__ float3 max(float3 a, float b) {
    return make_float3(max(a.x, b), max(a.y, b), max(a.z, b));
}

__device__ float3 abs(float3 a) {
    return make_float3(abs(a.x), abs(a.y), abs(a.z));
}

__device__ void atomicAddFloat3(float3* address, float3 val) {
    atomicAdd(&address->x, val.x);
    atomicAdd(&address->y, val.y);
    atomicAdd(&address->z, val.z);
}

__device__ float sdfSphere(float3 p, float r) {
    return length(p) - r;
}

__device__ float sdfBox(float3 p, float3 b) {
    float3 q = abs(p) - b;
    return length(max(q, 0.0f)) + min(max(q.x, max(q.y, q.z)), 0.0f);
}

__device__ bool CheckSphereSphere(Collider& a, Collider& b, float3& normal, float& depth) {
    float3 delta = a.position - b.position;
    float dist = length(delta);
    float radiiSum = a.dims.x + b.dims.x;

    if (dist < radiiSum && dist > 1e-6f) {
        normal = delta / dist;
        depth = radiiSum - dist;
        return true;
    }

    return false;
}

__device__ float3 CalculateColliderNormal(float3 p, Collider c) {
    float e = 0.001f;
    float3 n = make_float3(0, 0, 0);

    float3 localP = p - c.position;

    if (c.type == TYPE_SPHERE) {
        float len = length(localP);
        if (len < 1e-6f) return make_float3(0, 1, 0);
        return localP / length(localP);
    } else if (c.type == TYPE_BOX) {
        float d = sdfBox(localP, c.dims);
        float x = sdfBox(make_float3(localP.x + e, localP.y, localP.z), c.dims) - d;
        float y = sdfBox(make_float3(localP.x, localP.y + e, localP.z), c.dims) - d;
        float z = sdfBox(make_float3(localP.x, localP.y, localP.z + e), c.dims) - d;
        n = make_float3(x, y, z);
    }

    return n / length(n);
}

__device__ float3 GetRandomDir(uint32_t id) {
    float x = sinf(id * 12.9898f);
    float y = cosf(id * 78.233f);
    float z = sinf(id * 151.7182f);
    float len = sqrtf(x * x + y * y + z * z);
    return make_float3(x / len, y / len, z / len);
}

__device__ int3 PositionToCellCoord(float3 point, float radius) {
    int cellX = (int)floorf(point.x / radius);
    int cellY = (int)floorf(point.y / radius);
    int cellZ = (int)floorf(point.z / radius);
    return make_int3(cellX, cellY, cellZ);
}

__device__ uint32_t expandBits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ uint32_t HashCell(int cellX, int cellY, int cellZ) {
    // uint32_t a = (uint32_t)cellX * 15823;
    // uint32_t b = (uint32_t)cellY * 9737333;
    // uint32_t c = (uint32_t)cellZ * 440817757;
    // return a + b + c;
    return expandBits(cellX) | (expandBits(cellY) << 1) | (expandBits(cellZ) << 2);
}

__device__ uint32_t GetKeyFromHash(uint32_t hash, uint32_t hashTableSize) {
    return hash % hashTableSize;
}

__device__ float2 ConvertDensityToPressure(float density, float nearDensity, float targetDensity, float pressureMultiplier, float nearPressureMultiplier) {
    float densityError = density  - targetDensity;
    float pressure = densityError  * pressureMultiplier;
    pressure = max(pressure, 0.0f);
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

__device__ float SmoothingKernel(float dst, float radius, float scale) {
    if (dst >= radius) return 0.0f;
    float v = radius - dst;
    return v * v * scale;
}

__device__ float SmoothingKernelDerivative(float dst, float radius, float scale) {
    if (dst >= radius) return 0.0f;
    float v = radius - dst;
    return -v * scale;
}

__device__ float ViscositySmoothingKernel(float dst, float radius, float scale) {
    if (dst >= radius) return 0.0f;
    float v = radius * radius - dst * dst;
    return v * v * v * scale;
}

__device__ float NearDensityKernel(float dst, float radius, float scale) {
    if (dst >= radius) return 0.0f;
    float v = radius - dst;
    return v * v * v * scale;
}

__device__ float NearDensityDerivativeKernel(float dst, float radius, float scale) {
    if (dst >= radius) return 0.0f;
    float v = radius - dst;
    return -v * v * scale;
}

__device__ float2 CalculateDensity(const float* positions, const uint32_t* spatialIndices, const uint32_t* spatialKeys,
    const uint32_t* startIndices, int numParticles, uint32_t hashTableSize, float3 samplePoint, float smoothingRadius,
    float smoothingScale, float nearDensityScale
    ) {
    float density = 0.0f;
    float nearDensity = 0.0f;
    const float mass = 1.0f;

    int3 center = PositionToCellCoord(samplePoint, smoothingRadius);
    float sqrRadius = smoothingRadius * smoothingRadius;

    for (int offsetX = -1; offsetX <= 1; offsetX++) {
        for (int offsetY = -1; offsetY <= 1; offsetY++) {
            for (int offsetZ = -1; offsetZ <= 1; offsetZ++) {
                int3 cellCoord = make_int3(center.x + offsetX, center.y + offsetY, center.z + offsetZ);
                uint32_t hash = HashCell(cellCoord.x, cellCoord.y, cellCoord.z);
                uint32_t key = GetKeyFromHash(hash, hashTableSize);

                uint32_t startIndex = startIndices[key];

                for (uint32_t i = startIndex; i < numParticles; i++) {
                    if (spatialKeys[i] != key) break;

                    uint32_t particleIndex = spatialIndices[i];
                    uint32_t idx = particleIndex * 3;
                    float3 particlePos = make_float3(positions[idx], positions[idx + 1], positions[idx + 2]);

                    int3 neighborCell = PositionToCellCoord(particlePos, smoothingRadius);
                    uint32_t neighborHash = HashCell(neighborCell.x, neighborCell.y, neighborCell.z);
                    if (neighborHash != hash) continue;

                    float sqrDst = lengthSqr(particlePos - samplePoint);

                    if (sqrDst <= sqrRadius) {
                        float dst = sqrtf(sqrDst);
                        float influence = SmoothingKernel(dst, smoothingRadius, smoothingScale);
                        density += mass * influence;
                        nearDensity += mass * NearDensityKernel(dst, smoothingRadius, nearDensityScale);
                    }
                }
            }
        }

    }

    return make_float2(density, nearDensity);
}

__device__ float3 CalculatePressureForce(int particleIndex, float* positions, float* densities, float* nearDensities,
    uint32_t* spatialIndices, uint32_t* spatialKeys, uint32_t* startIndices, int numParticles, uint32_t hashTableSize,
    float smoothingRadius, float targetDensity, float pressureMultiplier,
    float nearPressureMultiplier, float smoothingDerivativeScale, float nearDerivativeScale) {
    float3 pressureForce = make_float3(0, 0, 0);
    const float mass = 1.0f;

    int currentIdx = particleIndex * 3;
    float3 samplePoint = make_float3(positions[currentIdx], positions[currentIdx + 1], positions[currentIdx + 2]);
    float density = densities[particleIndex];
    float nearDensity = nearDensities[particleIndex];

    int3 center = PositionToCellCoord(samplePoint, smoothingRadius);
    float sqrRadius = smoothingRadius * smoothingRadius;

    for (int offsetX = -1; offsetX <= 1; offsetX++) {
        for (int offsetY = -1; offsetY <= 1; offsetY++) {
            for (int offsetZ = -1; offsetZ <= 1; offsetZ++) {
                int3 cellCoord = make_int3(center.x + offsetX, center.y + offsetY, center.z + offsetZ);
                uint32_t key = GetKeyFromHash(HashCell(cellCoord.x, cellCoord.y, cellCoord.z), hashTableSize);
                uint32_t startIndex = startIndices[key];

                for (uint32_t i = startIndex; i < numParticles; i++) {
                    if (spatialKeys[i] != key) break;

                    uint32_t otherParticleIndex = spatialIndices[i];
                    if (particleIndex == otherParticleIndex) continue;

                    uint32_t otherIdx = otherParticleIndex * 3;
                    float px = __ldg(&positions[otherIdx]);
                    float py = __ldg(&positions[otherIdx + 1]);
                    float pz = __ldg(&positions[otherIdx + 2]);
                    float3 otherPos = make_float3(px, py, pz);

                    int3 neighborCell = PositionToCellCoord(otherPos, smoothingRadius);
                    uint32_t neighborHash = HashCell(neighborCell.x, neighborCell.y, neighborCell.z);
                    uint32_t currentHash = HashCell(cellCoord.x, cellCoord.y, cellCoord.z);

                    if (neighborHash != currentHash) continue;

                    float3 offset = otherPos - samplePoint;
                    float sqrDst = lengthSqr(offset);

                    if (sqrDst <= sqrRadius) {
                        // float dst = sqrtf(sqrDst);
                        // float3 dir = dst == 0.0f ? make_float3(0, 0, 0) : offset / dst;
                        float invDst = rsqrt(sqrDst);
                        float dst = sqrDst * invDst;
                        float3 dir = (sqrDst > 1e-10f) ? (offset * invDst) : make_float3(0, 0, 0);
                        float slope = SmoothingKernelDerivative(dst, smoothingRadius, smoothingDerivativeScale);
                        float nearSlope = NearDensityDerivativeKernel(dst, smoothingRadius, nearDerivativeScale);
                        float otherDensity = __ldg(&densities[otherParticleIndex]);
                        float otherNearDensity = __ldg(&nearDensities[otherParticleIndex]);
                        float2 sharedPressure = CalculateSharedPressure(density, nearDensity, otherDensity, otherNearDensity, targetDensity, pressureMultiplier, nearPressureMultiplier);

                        // float combinedForce = (sharedPressure.x * slope) + (sharedPressure.y * nearSlope);
                        // pressureForce += combinedForce * dir * mass / otherDensity;
                        pressureForce += dir * sharedPressure.x * slope * mass / otherDensity;
                        pressureForce += dir * sharedPressure.y * nearSlope * mass / otherDensity;
                    }
                }
            }
        }
    }

    return pressureForce;
}

__device__ float3 CalculateViscosityForce(int particleIndex, float* positions, float* velocities,
    uint32_t* spatialIndices, uint32_t* spatialKeys, uint32_t* startIndices,
    int numParticles, uint32_t hashTableSize, float smoothingRadius, float viscosityStrength, float viscosityScale) {
    float3 viscosityForce = make_float3(0, 0, 0);
    int currentIdx = particleIndex * 3;

    float3 position = make_float3(positions[currentIdx], positions[currentIdx + 1], positions[currentIdx + 2]);
    float3 velocity = make_float3(velocities[currentIdx], velocities[currentIdx + 1], velocities[currentIdx + 2]);

    int3 center = PositionToCellCoord(position, smoothingRadius);
    float sqrRadius = smoothingRadius * smoothingRadius;

    for (int offsetX = -1; offsetX <= 1; offsetX++) {
        for (int offsetY = -1; offsetY <= 1; offsetY++) {
            for (int offsetZ = -1; offsetZ <= 1; offsetZ++) {
                int3 cellCoord = make_int3(center.x + offsetX, center.y + offsetY, center.z + offsetZ);
                uint32_t key = GetKeyFromHash(HashCell(cellCoord.x, cellCoord.y, cellCoord.z), hashTableSize);
                uint32_t startIndex = startIndices[key];

                for (uint32_t i = startIndex; i < numParticles; i++) {
                    if (spatialKeys[i] != key) break;

                    uint32_t otherParticleIndex = spatialIndices[i];
                    if (particleIndex == otherParticleIndex) continue;

                    int otherIdx = otherParticleIndex * 3;
                    float3 otherPos = make_float3(positions[otherIdx], positions[otherIdx + 1], positions[otherIdx + 2]);

                    int3 neighborCell = PositionToCellCoord(otherPos, smoothingRadius);
                    uint32_t neighborHash = HashCell(neighborCell.x, neighborCell.y, neighborCell.z);
                    uint32_t currentHash = HashCell(cellCoord.x, cellCoord.y, cellCoord.z);
                    if (neighborHash != currentHash) continue;

                    float3 otherVel = make_float3(velocities[otherIdx], velocities[otherIdx + 1], velocities[otherIdx + 2]);

                    float sqrDst = lengthSqr(otherPos - position);

                    if (sqrDst <= sqrRadius) {
                        float dst = sqrtf(sqrDst);
                        float influence = ViscositySmoothingKernel(dst, smoothingRadius, viscosityScale);

                        viscosityForce += (otherVel - velocity) * influence;
                    }
                }
            }
        }
    }

    return viscosityForce * viscosityStrength;
}

__device__ void ResolveCollisions(float* positions, float* velocities, int numParticles, float particleSize, float boundsX, float boundsY, float boundsZ, float collisionDamping) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    int idx = i * 3;

    float3 posLocal = make_float3(positions[idx], positions[idx + 1], positions[idx + 2]);
    float3 velocityLocal = make_float3(velocities[idx], velocities[idx + 1], velocities[idx + 2]);

    const float3 halfSize = make_float3(boundsX / 2, boundsY / 2, boundsZ / 2);
    const float3 edgeDst = make_float3(halfSize.x - abs(posLocal.x), halfSize.y - abs(posLocal.y), halfSize.z - abs(posLocal.z));

    if (edgeDst.x <= 0) {
        posLocal.x = halfSize.x * sign(posLocal.x);
        velocityLocal.x *= -1 * collisionDamping;
    }
    if (edgeDst.y <= 0) {
        posLocal.y = halfSize.y * sign(posLocal.y);
        velocityLocal.y *= -1 * collisionDamping;
    }
    if (edgeDst.z <= 0) {
        posLocal.z = halfSize.z * sign(posLocal.z);
        velocityLocal.z *= -1 * collisionDamping;
    }

    positions[idx] = posLocal.x;
    positions[idx + 1] = posLocal.y;
    positions[idx + 2] = posLocal.z;

    velocities[idx] = velocityLocal.x;
    velocities[idx + 1] = velocityLocal.y;
    velocities[idx + 2] = velocityLocal.z;
}

__device__ float3 CalculateCombinedForces(
    int particleIndex,
    float* positions,
    float* velocities,
    float* densities,
    float* nearDensities,
    uint32_t* spatialIndices,
    uint32_t* spatialKeys,
    uint32_t* startIndices,
    int numParticles,
    uint32_t hashTableSize,
    float smoothingRadius,
    float targetDensity,
    float pressureMultiplier,
    float nearPressureMultiplier,
    float viscosityStrength,
    float smoothingDerivativeScale,
    float nearDerivativeScale,
    float viscosityScale
) {
    float3 totalForce = make_float3(0, 0, 0);
    const float mass = 1.0f;

    int currentIdx = particleIndex * 3;
    float3 samplePoint = make_float3(positions[currentIdx], positions[currentIdx + 1], positions[currentIdx + 2]);
    float3 velocity = make_float3(velocities[currentIdx], velocities[currentIdx + 1], velocities[currentIdx + 2]);

    float density = densities[particleIndex];
    float nearDensity = nearDensities[particleIndex];

    int3 center = PositionToCellCoord(samplePoint, smoothingRadius);
    float sqrRadius = smoothingRadius * smoothingRadius;

    for (int offsetX = -1; offsetX <= 1; offsetX++) {
        for (int offsetY = -1; offsetY <= 1; offsetY++) {
            for (int offsetZ = -1; offsetZ <= 1; offsetZ++) {
                int3 cellCoord = make_int3(center.x + offsetX, center.y + offsetY, center.z + offsetZ);
                uint32_t key = GetKeyFromHash(HashCell(cellCoord.x, cellCoord.y, cellCoord.z), hashTableSize);
                uint32_t startIndex = startIndices[key];

                for (uint32_t i = startIndex; i < numParticles; i++) {
                    if (spatialKeys[i] != key) break;

                    uint32_t otherParticleIndex = spatialIndices[i];
                    if (particleIndex == otherParticleIndex) continue;

                    uint32_t otherIdx = otherParticleIndex * 3;

                    // Load position once for both calculations
                    float px = __ldg(&positions[otherIdx]);
                    float py = __ldg(&positions[otherIdx + 1]);
                    float pz = __ldg(&positions[otherIdx + 2]);
                    float3 otherPos = make_float3(px, py, pz);

                    float3 offset = otherPos - samplePoint;
                    float sqrDst = lengthSqr(offset);

                    if (sqrDst <= sqrRadius) {
                        float invDst = rsqrt(sqrDst);
                        float dst = sqrDst * invDst;

                        float3 dir = (sqrDst > 1e-10f) ? (offset * invDst) : make_float3(0, 0, 0);
                        float slope = SmoothingKernelDerivative(dst, smoothingRadius, smoothingDerivativeScale);
                        float nearSlope = NearDensityDerivativeKernel(dst, smoothingRadius, nearDerivativeScale);

                        float otherDensity = __ldg(&densities[otherParticleIndex]);
                        float otherNearDensity = __ldg(&nearDensities[otherParticleIndex]);

                        float2 sharedPressure = CalculateSharedPressure(density, nearDensity, otherDensity, otherNearDensity, targetDensity, pressureMultiplier, nearPressureMultiplier);

                        totalForce += dir * sharedPressure.x * slope * mass / otherDensity;
                        totalForce += dir * sharedPressure.y * nearSlope * mass / otherDensity;

                        // Only load velocity if we are actually inside the radius
                        float vx = __ldg(&velocities[otherIdx]);
                        float vy = __ldg(&velocities[otherIdx + 1]);
                        float vz = __ldg(&velocities[otherIdx + 2]);
                        float3 otherVel = make_float3(vx, vy, vz);

                        float influence = ViscositySmoothingKernel(dst, smoothingRadius, viscosityScale);
                        totalForce += (otherVel - velocity) * influence * viscosityStrength;
                    }
                }
            }
        }
    }

    return totalForce;
}

__global__ void ApplyPressureForces_Optimized(
    const float* __restrict__ predictedPositions,
    float*                    velocities,
    const float* __restrict__ densities,
    const float* __restrict__ nearDensities,
    const uint32_t* __restrict__ spatialIndices,
    const uint32_t* __restrict__ spatialKeys,
    const uint32_t* __restrict__ startIndices,
    int numParticles, uint32_t hashTableSize,
    float smoothingRadius, float targetDensity,
    float pressureMultiplier, float nearPressureMultiplier,
    float viscosityStrength, float dt,
    float smoothingDerivativeScale, float viscosityScale, float nearDerivativeScale
) {
    const int tid = threadIdx.x;
    const int i   = blockIdx.x * blockDim.x + tid;
    const int BS  = blockDim.x;

    extern __shared__ float s_data[];
    float* s_posX = s_data;
    float* s_posY = s_posX + BS;
    float* s_posZ = s_posY + BS;
    float* s_velX = s_posZ + BS;
    float* s_velY = s_velX + BS;
    float* s_velZ = s_velY + BS;
    float* s_den  = s_velZ + BS;
    float* s_nden = s_den  + BS;

    float3 samplePos, myVel;
    float  myDensity, myNearDensity;

    if (i < numParticles) {
        int idx   = i * 3;
        samplePos     = make_float3(__ldg(&predictedPositions[idx]),
                                    __ldg(&predictedPositions[idx+1]),
                                    __ldg(&predictedPositions[idx+2]));
        myVel         = make_float3(__ldg(&velocities[idx]),
                                    __ldg(&velocities[idx+1]),
                                    __ldg(&velocities[idx+2]));
        myDensity     = __ldg(&densities[i]);
        myNearDensity = __ldg(&nearDensities[i]);
    } else {
        samplePos = myVel = make_float3(0.f, 0.f, 0.f);
        myDensity = 1.f; myNearDensity = 0.f;
    }

    s_posX[tid] = samplePos.x;
    s_posY[tid] = samplePos.y;
    s_posZ[tid] = samplePos.z;
    s_velX[tid] = myVel.x;
    s_velY[tid] = myVel.y;
    s_velZ[tid] = myVel.z;
    s_den [tid] = myDensity;
    s_nden[tid] = myNearDensity;
    __syncthreads();

    float3 pressureForce  = make_float3(0.f, 0.f, 0.f);
    float3 viscosityForce = make_float3(0.f, 0.f, 0.f);
    const float sqrRadius = smoothingRadius * smoothingRadius;

    if (i >= numParticles) return;

    // After the Morton-code sort, particles in the same block are
    // spatially clustered and are each other's most likely neighbors.
    // We iterate all BS slots from smem (zero gmem traffic).
    {
        int blockStart = blockIdx.x * BS;
        #pragma unroll 4
        for (int j = 0; j < BS; j++) {
            int globalJ = blockStart + j;
            if (globalJ >= numParticles || globalJ == i) continue;

            float3 otherPos = make_float3(s_posX[j], s_posY[j], s_posZ[j]);
            float3 offset   = otherPos - samplePos;
            float  sqrDst   = lengthSqr(offset);

            if (sqrDst <= sqrRadius && sqrDst > 1e-20f) {
                float invDst = rsqrtf(sqrDst);
                float dst    = sqrDst * invDst;
                float3 dir   = offset * invDst;

                float otherDensity     = s_den[j];
                float otherNearDensity = s_nden[j];
                float3 otherVel        = make_float3(s_velX[j], s_velY[j], s_velZ[j]);

                float slope     = SmoothingKernelDerivative(dst, smoothingRadius, smoothingDerivativeScale);
                float nearSlope = NearDensityDerivativeKernel(dst, smoothingRadius, nearDerivativeScale);
                float2 sp       = CalculateSharedPressure(myDensity, myNearDensity,
                                    otherDensity, otherNearDensity,
                                    targetDensity, pressureMultiplier, nearPressureMultiplier);
                float invOtherDen = 1.f / otherDensity;
                pressureForce += dir * (sp.x * slope     * invOtherDen);
                pressureForce += dir * (sp.y * nearSlope * invOtherDen);

                float influence = ViscositySmoothingKernel(dst, smoothingRadius, viscosityScale);
                viscosityForce += (otherVel - myVel) * influence;
            }
        }
    }

    // Each thread independently walks its own 27-cell neighborhood.
    // Particles belonging to this block are skipped (already handled).
    // __ldg() routes through the read-only cache (ld.global.nc),
    // improving reuse across threads that share neighbors.
    {
        int3     center     = PositionToCellCoord(samplePos, smoothingRadius);
        uint32_t blockStart = (uint32_t)(blockIdx.x * BS);
        uint32_t blockEnd   = blockStart + (uint32_t)BS;

        for (int c = 0; c < 27; c++) {
            int3     cellCoord  = center + CELL_OFFSETS[c];
            uint32_t key        = GetKeyFromHash(HashCell(cellCoord.x, cellCoord.y, cellCoord.z), hashTableSize);
            uint32_t startIndex = __ldg(&startIndices[key]);
            if (startIndex == 0xffffffff) continue;

            for (uint32_t j = startIndex; j < (uint32_t)numParticles; j++) {
                if (__ldg(&spatialKeys[j]) != key) break;
                if (j >= blockStart && j < blockEnd) continue; // handled in Phase 1
                if (j == (uint32_t)i) continue;

                int    oidx     = j * 3;
                float3 otherPos = make_float3(__ldg(&predictedPositions[oidx]),
                                              __ldg(&predictedPositions[oidx+1]),
                                              __ldg(&predictedPositions[oidx+2]));
                float3 offset   = otherPos - samplePos;
                float  sqrDst   = lengthSqr(offset);

                if (sqrDst <= sqrRadius && sqrDst > 1e-20f) {
                    float invDst = rsqrtf(sqrDst);
                    float dst    = sqrDst * invDst;
                    float3 dir   = offset * invDst;

                    float otherDensity     = __ldg(&densities[j]);
                    float otherNearDensity = __ldg(&nearDensities[j]);
                    float3 otherVel        = make_float3(__ldg(&velocities[oidx]),
                                                         __ldg(&velocities[oidx+1]),
                                                         __ldg(&velocities[oidx+2]));

                    float slope     = SmoothingKernelDerivative(dst, smoothingRadius, smoothingDerivativeScale);
                    float nearSlope = NearDensityDerivativeKernel(dst, smoothingRadius, nearDerivativeScale);
                    float2 sp       = CalculateSharedPressure(myDensity, myNearDensity,
                                        otherDensity, otherNearDensity,
                                        targetDensity, pressureMultiplier, nearPressureMultiplier);
                    float invOtherDen = 1.f / otherDensity;
                    pressureForce += dir * (sp.x * slope     * invOtherDen);
                    pressureForce += dir * (sp.y * nearSlope * invOtherDen);

                    float influence = ViscositySmoothingKernel(dst, smoothingRadius, viscosityScale);
                    viscosityForce += (otherVel - myVel) * influence;
                }
            }
        }
    }

    float3 totalForce   = pressureForce + (viscosityForce * viscosityStrength);
    float3 acceleration = totalForce / max(myDensity, 0.0001f);

    int idx = i * 3;
    velocities[idx]   += acceleration.x * dt;
    velocities[idx+1] += acceleration.y * dt;
    velocities[idx+2] += acceleration.z * dt;
}


__global__ void UpdatePositions(float* positions, float* velocities, int numParticles, float particleSize, float boundsX, float boundsY, float boundsZ,
    float collisionDamping, float gravity, float dt, Collider* colliders, int numColliders, float smoothingRadius, float colliderDragModifier) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    int idx = i * 3;

    positions[idx] += velocities[idx] * dt;
    positions[idx + 1] += velocities[idx + 1] * dt;
    positions[idx + 2] += velocities[idx + 2] * dt;

    float3 posLocal = make_float3(positions[idx], positions[idx + 1], positions[idx + 2]);
    float3 velocityLocal = make_float3(velocities[idx], velocities[idx + 1], velocities[idx + 2]);

    // Primitive collision
    for (int k = 0; k < numColliders; k++) {
        Collider col = colliders[k];
        float3 relPos = posLocal - col.position;
        float dist = 0.0f;

        if (col.type == TYPE_SPHERE) {
            dist = sdfSphere(relPos, col.dims.x);
        } else if (col.type == TYPE_BOX) {
            dist = sdfBox(relPos, col.dims);
        }

        if (col.isDynamic && dist < smoothingRadius) {
            float3 relativeVel = velocityLocal - col.velocity;
            float weight = 1.0f - (max(0.0f, dist) / smoothingRadius);

            float3 dragForce = relativeVel * weight * colliderDragModifier;
            atomicAddFloat3(&colliders[k].forceAccumulator, dragForce);
            velocityLocal -= dragForce * dt;

            float particleMass = 1.0f;
            float3 buoyancyDirection = make_float3(0, 1, 0);

            float buoyancyStrength = weight * particleMass * gravity;
            float3 buoyancyForce = buoyancyDirection * buoyancyStrength;
            atomicAddFloat3(&colliders[k].forceAccumulator, buoyancyForce);
        }

        if (dist < particleSize) {
            float3 normal = CalculateColliderNormal(posLocal, col);
            float penetration = particleSize - dist;

            posLocal += normal * penetration;

            float3 relativeVelocity = velocityLocal - col.velocity;
            float normalVel = dot(relativeVelocity, normal);
            if (normalVel < 0) {
                float3 velocityChange = normal * normalVel * (1.0f + collisionDamping);
                velocityLocal -= velocityChange;

                if (col.isDynamic) {
                    float particleMass = 3.0f;
                    float3 impulse = velocityChange * particleMass;
                    atomicAddFloat3(&colliders[k].forceAccumulator, impulse / dt);

                    float stiffness = 13000.0f;
                    float3 reactionForce = normal * penetration * stiffness;
                    atomicAddFloat3(&colliders[k].forceAccumulator, -1.0f * reactionForce);
                }
            }
        }
    }

    const float3 halfSize = make_float3(boundsX / 2, boundsY / 2, boundsZ / 2);
    const float3 edgeDst = make_float3(
        halfSize.x - std::abs(posLocal.x),
        halfSize.y - std::abs(posLocal.y),
        halfSize.z - std::abs(posLocal.z)
        );

    if (edgeDst.x <= 0) {
        posLocal.x = posLocal.x > 0 ? halfSize.x : -halfSize.x;
        velocityLocal.x *= -1 * collisionDamping;
    }
    if (edgeDst.y <= 0) {
        posLocal.y = posLocal.y > 0 ? halfSize.y : -halfSize.y;
        velocityLocal.y *= -1 * collisionDamping;
    }
    if (edgeDst.z <= 0) {
        posLocal.z = posLocal.z > 0 ? halfSize.z : -halfSize.z;
        velocityLocal.z *= -1 * collisionDamping;
    }

    positions[idx] = posLocal.x;
    positions[idx + 1] = posLocal.y;
    positions[idx + 2] = posLocal.z;

    velocities[idx] = velocityLocal.x;
    velocities[idx + 1] = velocityLocal.y;
    velocities[idx + 2] = velocityLocal.z;
}

__global__ void IntegrateColliders(Collider* colliders, int numColliders, float boundsX, float boundsY, float boundsZ,
    float gravity, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numColliders) return;

    Collider& col = colliders[i];

    if (!col.isDynamic) {
        col.forceAccumulator = make_float3(0, 0, 0);
        return;
    }

    float3 totalForce = col.forceAccumulator;
    totalForce.y += -1.0f * gravity * col.mass;

    float3 acceleration = totalForce / col.mass;
    col.velocity += acceleration * dt;

    col.position += col.velocity * dt;

    const float3 halfSize = make_float3(boundsX / 2, boundsY / 2, boundsZ / 2);

    float3 extents;
    if (col.type == TYPE_SPHERE) {
        extents = make_float3(col.dims.x, col.dims.x, col.dims.x);
    } else {
        extents = col.dims;
    }

    if (col.position.x - extents.x < -halfSize.x) {
        col.position.x = -halfSize.x + extents.x;
        col.velocity.x *= -0.5f;
    }
    if (col.position.x + extents.x > halfSize.x) {
        col.position.x = halfSize.x - extents.x;
        col.velocity.x *= -0.5f;
    }

    // Y Axis
    if (col.position.y - extents.y < -halfSize.y) {
        col.position.y = -halfSize.y + extents.y;
        col.velocity.y *= -0.5f;
    }
    if (col.position.y + extents.y > halfSize.y) {
        col.position.y = halfSize.y - extents.y;
        col.velocity.y *= -0.5f;
    }

    // Z Axis
    if (col.position.z - extents.z < -halfSize.z) {
        col.position.z = -halfSize.z + extents.z;
        col.velocity.z *= -0.5f;
    }
    if (col.position.z + extents.z > halfSize.z) {
        col.position.z = halfSize.z - extents.z;
        col.velocity.z *= -0.5f;
    }

    col.forceAccumulator = make_float3(0, 0, 0);
}

__global__ void ResolveColliderCollisions(Collider* colliders, int numColliders) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numColliders) return;

    // Iterate against all other colliders with index > i to avoid double processing
    for (int j = i + 1; j < numColliders; j++) {
        Collider& colA = colliders[i];
        Collider& colB = colliders[j];

        // Skip if both are static
        if (!colA.isDynamic && !colB.isDynamic) continue;

        float3 normal = make_float3(0,1,0);
        float depth = 0.0f;
        bool collision = false;

        // Dispatch based on types
        if (colA.type == TYPE_SPHERE && colB.type == TYPE_SPHERE) {
            collision = CheckSphereSphere(colA, colB, normal, depth);
        }
        // else if (colA.type == TYPE_SPHERE && colB.type == TYPE_BOX) {
        //     collision = CheckSphereBox(colA, colB, normal, depth);
        // }
        // else if (colA.type == TYPE_BOX && colB.type == TYPE_SPHERE) {
        //     // Flip normal because we pass (Sphere, Box)
        //     collision = CheckSphereBox(colB, colA, normal, depth);
        //     normal = normal * -1.0f;
        // }
        // else if (colA.type == TYPE_BOX && colB.type == TYPE_BOX) {
        //     collision = CheckBoxBox(colA, colB, normal, depth);
        // }

        if (collision) {
            const float percent = 0.2f; // Penetration percentage to correct
            const float slop = 0.001f;  // Penetration allowance
            float correctionMag = max(depth - slop, 0.0f) * percent;

            float invMassA = colA.isDynamic ? 1.0f / colA.mass : 0.0f;
            float invMassB = colB.isDynamic ? 1.0f / colB.mass : 0.0f;
            float invMassSum = invMassA + invMassB;

            if (invMassSum == 0.0f) continue;

            float3 correction = normal * (correctionMag / invMassSum);

            if (colA.isDynamic) colA.position += correction * invMassA;
            if (colB.isDynamic) colB.position -= correction * invMassB;

            float3 relVel = colA.velocity - colB.velocity;
            float velAlongNormal = dot(relVel, normal);

            // Do not resolve if velocities are separating
            if (velAlongNormal > 0) continue;

            float restitution = 0.5f; // Bounciness
            float j = -(1.0f + restitution) * velAlongNormal;
            j /= invMassSum;

            float3 impulse = normal * j;

            if (colA.isDynamic) colA.velocity += impulse * invMassA;
            if (colB.isDynamic) colB.velocity -= impulse * invMassB;
        }
    }
}

__global__ void PredictPositions(float* positions, float* predictedPositions, float* velocities, int numParticles, float gravity, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    int idx = i * 3;

    velocities[idx + 1] += -1 * gravity * dt;
    predictedPositions[idx] = positions[idx] + velocities[idx] * dt;
    predictedPositions[idx + 1] = positions[idx + 1] + velocities[idx + 1] * dt;
    predictedPositions[idx + 2] = positions[idx + 2] + velocities[idx + 2] * dt;
    // predictedPositions[idx] = positions[idx] + velocities[idx] * 1 / 120.0f;
    // predictedPositions[idx + 1] = positions[idx + 1] + velocities[idx + 1] * 1 / 120.0f;
    // predictedPositions[idx + 2] = positions[idx + 2] + velocities[idx + 2] * 1 / 120.0f;
}

__global__ void UpdateDensities(float* positions, float* velocities, float* densities, float* nearDensities,
    uint32_t* spatialIndices, uint32_t* spatialKeys, uint32_t* startIndices, int numParticles, uint32_t hashTableSize,
    float smoothingRadius, float gravity, float dt, float smoothingScale, float nearDensityScale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    int idx = i * 3;

    float3 samplePoint  = make_float3(positions[idx], positions[idx + 1], positions[idx + 2]);
    float2 result = CalculateDensity(positions, spatialIndices, spatialKeys, startIndices, numParticles, hashTableSize, samplePoint, smoothingRadius, smoothingScale, nearDensityScale);
    densities[i] = result.x;
    nearDensities[i] = result.y;
}

__global__ void UpdateDensities_Shared(
    float* positions, // SORTED
    uint32_t* spatialIndices,
    uint32_t* spatialKeys,
    uint32_t* startIndices,
    int numParticles,
    uint32_t hashTableSize,
    float smoothingRadius,
    float smoothingScale,
    float nearDensityScale,
    float* densities,
    float* nearDensities
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    int idx = i * 3;
    float3 myPos = make_float3(positions[idx], positions[idx+1], positions[idx+2]);

    extern __shared__ float3 s_pos[];
    s_pos[threadIdx.x] = myPos;
    __syncthreads();

    float density = 0.0f;
    float nearDensity = 0.0f;
    float sqrRadius = smoothingRadius * smoothingRadius;
    const float mass = 1.0f;

    #pragma unroll
    for (int j = 0; j < blockDim.x; j++) {
        if ((blockIdx.x * blockDim.x + j) >= numParticles) break;
        float3 sPos = s_pos[j];
        float sqrDst = lengthSqr(sPos - myPos);
        if (sqrDst <= sqrRadius) {
            float dst = sqrtf(sqrDst);
            density += mass * SmoothingKernel(dst, smoothingRadius, smoothingScale);
            nearDensity += mass * NearDensityKernel(dst, smoothingRadius, nearDensityScale);
        }
    }

    int3 center = PositionToCellCoord(myPos, smoothingRadius);
    uint32_t blockStart = blockIdx.x * blockDim.x;
    uint32_t blockEnd   = blockStart + blockDim.x;

    for (int offsetX = -1; offsetX <= 1; offsetX++) {
        for (int offsetY = -1; offsetY <= 1; offsetY++) {
            for (int offsetZ = -1; offsetZ <= 1; offsetZ++) {
                int3 cellCoord = make_int3(center.x + offsetX, center.y + offsetY, center.z + offsetZ);
                uint32_t hash = HashCell(cellCoord.x, cellCoord.y, cellCoord.z);
                uint32_t key = GetKeyFromHash(hash, hashTableSize);
                uint32_t startIndex = startIndices[key];

                if (startIndex == 0xffffffff) continue;

                for (uint32_t j = startIndex; j < numParticles; j++) {
                    if (spatialKeys[j] != key) break;
                    if (j >= blockStart && j < blockEnd) continue; // Skip shared mem

                    int oIdx = j * 3;
                    float px = __ldg(&positions[oIdx]);
                    float py = __ldg(&positions[oIdx + 1]);
                    float pz = __ldg(&positions[oIdx + 2]);
                    float3 otherPos = make_float3(px, py, pz);

                    float sqrDst = lengthSqr(otherPos - myPos);
                    if (sqrDst <= sqrRadius) {
                        float dst = sqrtf(sqrDst);
                        density += mass * SmoothingKernel(dst, smoothingRadius, smoothingScale);
                        nearDensity += mass * NearDensityKernel(dst, smoothingRadius, nearDensityScale);
                    }
                }
            }
        }
    }

    densities[i] = density;
    nearDensities[i] = nearDensity;
}

__global__ void UpdateSpatialHash(float* positions, int numParticles, uint32_t hashTableSize, float radius, uint32_t* spatialIndices, uint32_t* spatialKeys, uint32_t* startIndices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    int idx = i * 3;
    float3 pos = make_float3(positions[idx], positions[idx + 1], positions[idx + 2]);

    // startIndices[i] = 0xffffffff;

    int3 cell = PositionToCellCoord(pos, radius);
    uint32_t cellKey = GetKeyFromHash(HashCell(cell.x, cell.y, cell.z), hashTableSize);

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

__global__ void SortData(
    int numParticles,
    uint32_t* spatialIndices,
    float* positions, float* sortedPositions,
    float* velocities, float* sortedVelocities
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    int originalIndex = spatialIndices[i];
    int idx = i * 3;
    int originalIdx = originalIndex * 3;

    sortedPositions[idx] = positions[originalIdx];
    sortedPositions[idx + 1] = positions[originalIdx + 1];
    sortedPositions[idx + 2] = positions[originalIdx + 2];

    sortedVelocities[idx] = velocities[originalIdx];
    sortedVelocities[idx + 1] = velocities[originalIdx + 1];
    sortedVelocities[idx + 2] = velocities[originalIdx + 2];
}

__global__ void ReorderVelocities(
    int numParticles,
    uint32_t* spatialIndices,
    float* sortedVelocities,
    float* originalVelocities
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    int originalIndex = spatialIndices[i];
    int idx = i * 3;
    int originalIdx = originalIndex * 3;

    originalVelocities[originalIdx] = sortedVelocities[idx];
    originalVelocities[originalIdx + 1] = sortedVelocities[idx + 1];
    originalVelocities[originalIdx + 2] = sortedVelocities[idx + 2];
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

    cudaMemset(d_spatialIndices, 0xffffffff, m_maxParticles * sizeof(uint32_t));
    cudaMemset(d_startIndices, 0xffffffff, m_hashTableSize * sizeof(uint32_t));

    UpdateSpatialHash<<<numBlock, blockSize>>>(d_predictedPositions, m_numParticles, m_hashTableSize, m_params.smoothingRadius, d_spatialIndices, d_spatialKeys, d_startIndices);
    cudaDeviceSynchronize();

    // Sort by cell key
    thrust::device_ptr<uint32_t> t_keys(d_spatialKeys);
    thrust::device_ptr<uint32_t> t_indices(d_spatialIndices);

    thrust::sort_by_key(t_keys, t_keys + m_numParticles, t_indices);

    UpdateStartIndices<<<numBlock, blockSize>>>(d_spatialKeys, d_startIndices, m_numParticles);
    cudaDeviceSynchronize();
}

SPHSolver::SPHSolver(int maxParticles) : m_maxParticles(maxParticles), m_numParticles(0) {
    m_hashTableSize = m_maxParticles * 2;
    size_t size = m_maxParticles * 3 * sizeof(float);
    cudaMalloc(&d_positions, size);
    cudaMalloc(&d_predictedPositions, size);
    cudaMalloc(&d_sortedPredictedPositions, size);
    cudaMalloc(&d_velocities, size);
    cudaMalloc(&d_sortedVelocities, size);
    cudaMalloc(&d_densities, m_maxParticles * sizeof(float));
    cudaMalloc(&d_nearDensities, m_maxParticles * sizeof(float));
    cudaMalloc(&d_spatialIndices, m_maxParticles * sizeof(uint32_t));
    cudaMalloc(&d_spatialKeys, m_maxParticles * sizeof(uint32_t));
    cudaMalloc(&d_startIndices, m_hashTableSize * sizeof(uint32_t));
    cudaMalloc(&d_colliders, 10 * sizeof(Collider));
}

SPHSolver::~SPHSolver() {
    if (d_positions) cudaFree(d_positions);
    if (d_velocities) cudaFree(d_velocities);
}

void SPHSolver::init(const std::vector<float> &positions, const std::vector<float> &velocities) {
    m_numParticles = positions.size() / 3;
    if (m_numParticles > m_maxParticles) m_numParticles = m_maxParticles;

    size_t copySize = m_numParticles * 3 * sizeof(float);
    cudaMemcpy(d_positions, positions.data(), copySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocities, velocities.data(), copySize, cudaMemcpyHostToDevice);

    float h = m_params.smoothingRadius;
    float h5 = powf(h, 5.0f);
    float h6 = powf(h, 6.0f);
    float h9 = powf(h, 9.0f);
    float pi = std::numbers::pi_v<float>;

    m_params.densityScale = 15.0f / (2.0f * pi * h5);
    m_params.pressureScale = 15.0f / (pi * h5);
    m_params.viscosityScale = 315.0f / (64.0f * pi * h9);
    m_params.nearDensityScale = 15.0f / (pi * h6);
    m_params.nearPressureScale = 45.0f / (pi * h6);
}

void SPHSolver::update(float dt) {
    int blockSize = 256;
    int numBlock = (m_numParticles + blockSize - 1) / blockSize;

    // Apply gravity and predict next positions
    PredictPositions<<<numBlock, blockSize>>>(d_positions, d_predictedPositions, d_velocities,
        m_numParticles, m_params.gravity, dt);
    // cudaDeviceSynchronize();

    UpdateSpatialLookup();

    SortData<<<numBlock, blockSize>>>(m_numParticles, d_spatialIndices, d_predictedPositions, d_sortedPredictedPositions, d_velocities, d_sortedVelocities);

    // Calculate and apply densities
    size_t smemDensity = blockSize * sizeof(float3);
    // UpdateDensities<<<numBlock, blockSize>>>(d_predictedPositions, d_velocities, d_densities, d_nearDensities,
    //     d_spatialIndices, d_spatialKeys, d_startIndices, m_numParticles, m_hashTableSize,
    //     m_params.smoothingRadius, m_params.gravity, dt, m_params.densityScale, m_params.nearDensityScale);
    UpdateDensities_Shared<<<numBlock, blockSize, smemDensity>>>(
         d_sortedPredictedPositions,
         d_spatialIndices, d_spatialKeys, d_startIndices, m_numParticles, m_hashTableSize,
         m_params.smoothingRadius, m_params.densityScale, m_params.nearDensityScale,
         d_densities, d_nearDensities
     );
    // cudaDeviceSynchronize();

    // Calculate and apply pressure forces
    // ApplyPressureForces<<<numBlock, blockSize>>>(d_predictedPositions, d_velocities, d_densities, d_nearDensities,
    //     d_spatialIndices, d_spatialKeys, d_startIndices, m_numParticles, m_hashTableSize,
    //     m_params.smoothingRadius, m_params.targetDensity, m_params.pressureMultiplier, m_params.nearPressureMultiplier, m_params.viscosityStrength, dt, m_params.pressureScale, m_params.viscosityScale, m_params.nearPressureScale);
    size_t smemPressure = blockSize * 16 * sizeof(float);
    ApplyPressureForces_Optimized<<<numBlock, blockSize, smemPressure>>>(
        d_sortedPredictedPositions, // Sorted
        d_sortedVelocities,         // Sorted (This gets updated!)
        d_densities,                // Calculated from sorted data (matches index)
        d_nearDensities,            // Calculated from sorted data (matches index)
        d_spatialIndices, d_spatialKeys, d_startIndices, m_numParticles, m_hashTableSize,
        m_params.smoothingRadius, m_params.targetDensity, m_params.pressureMultiplier,
        m_params.nearPressureMultiplier, m_params.viscosityStrength, dt,
        m_params.pressureScale, m_params.viscosityScale, m_params.nearPressureScale
    );
    // cudaDeviceSynchronize();

    ReorderVelocities<<<numBlock, blockSize>>>(m_numParticles, d_spatialIndices, d_sortedVelocities, d_velocities);
    cudaDeviceSynchronize();

    // Update positions and handle collisions
    UpdatePositions<<<numBlock, blockSize>>>(d_positions, d_velocities, m_numParticles,
        m_params.particleSize, m_params.boundsX, m_params.boundsY, m_params.boundsZ,
        m_params.collisionDamping, m_params.gravity, dt, d_colliders, m_numColliders, m_params.smoothingRadius, m_params.colliderDragMultiplier);
    // cudaDeviceSynchronize();

    IntegrateColliders<<<1, 32>>>(d_colliders, m_numColliders, m_params.boundsX, m_params.boundsY, m_params.boundsZ, m_params.gravity, dt);
    // cudaDeviceSynchronize();

    ResolveColliderCollisions<<<1, m_numColliders>>>(d_colliders, m_numColliders);
    cudaDeviceSynchronize();
}

void SPHSolver::addCollider(Collider collider) {
    m_colliders.push_back(collider);
    m_numColliders = m_colliders.size();

    cudaMemcpy(d_colliders, m_colliders.data(), m_numColliders * sizeof(Collider), cudaMemcpyHostToDevice);
}

void SPHSolver::getColliders(std::vector<Collider> &outColliders) {
    if (m_numColliders == 0) return;
    outColliders.resize(m_numColliders);
    cudaMemcpy(outColliders.data(), d_colliders, m_numColliders * sizeof(Collider), cudaMemcpyDeviceToHost);
}

void SPHSolver::getPositions(float* outPositions) {
    // cudaMemcpy(outPositions, d_positions, m_numParticles * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(outPositions, d_positions, m_numParticles * 3 * sizeof(float), cudaMemcpyHostToDevice);
}

void SPHSolver::setParams(const SPHParams &params) {
    m_params = params;
}
