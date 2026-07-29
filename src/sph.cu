#include <sph.cuh>

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

__global__ void soa_to_aos(float* aos, const float* x, const float* y, const float* z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    aos[i * 3] = x[i];
    aos[i * 3 + 1] = y[i];
    aos[i * 3 + 2] = z[i];
}

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

__device__ float3 max(float3 a, float3 b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__device__ float3 abs(float3 a) {
    return make_float3(abs(a.x), abs(a.y), abs(a.z));
}


__device__ void atomicAddFloat3(float3* address, float3 val) {
    atomicAdd(&address->x, val.x);
    atomicAdd(&address->y, val.y);
    atomicAdd(&address->z, val.z);
}

__device__ uint32_t randomState(float3 pos, float simTime) {
    uint32_t ux = __float_as_uint(pos.x);
    uint32_t uy = __float_as_uint(pos.y);
    uint32_t uz = __float_as_uint(pos.z);
    uint32_t ut = __float_as_uint(simTime);

    return ux * 19349669u
         + uy * 83492837u
         + uz * 73856131u
         + ut * 4785773u;
}

__device__ float randomValue(uint32_t& rngState) {
    rngState ^= rngState << 13;
    rngState ^= rngState >> 17;
    rngState ^= rngState << 5;
    return (rngState & 0x00FFFFFFu) / (float)0x01000000u;
}

__device__ float lerpf(float a, float b, float t) {
    return a + t * (b - a);
}

__device__ float sdfSphere(float3 p, float r) {
    return length(p) - r;
}

__device__ float sdfBox(float3 p, float3 b) {
    float3 q = abs(p) - b;
    return length(max(q, 0.0f)) + min(max(q.x, max(q.y, q.z)), 0.0f);
}

__device__ float sdfMesh(float3 localPos, Collider& col) {
    float3 minB = col.gridMinBounds;
    float3 maxB = col.gridMaxBounds;

    // Check if the particle is outside the mesh's bounding box
    if (localPos.x < minB.x || localPos.x > maxB.x ||
        localPos.y < minB.y || localPos.y > maxB.y ||
        localPos.z < minB.z || localPos.z > maxB.z) {

        // Calculate the accurate positive distance to the outside of the box
        float3 q = max(minB - localPos, max(localPos - maxB, make_float3(0.0f, 0.0f, 0.0f)));
        return length(q) + 0.1f; // The +0.1f acts as a safety buffer
        }

    // Map local coordinate to 3D texture coordinate [0.0, 1.0]
    float u = (localPos.x - minB.x) / (maxB.x - minB.x);
    float v = (localPos.y - minB.y) / (maxB.y - minB.y);
    float w = (localPos.z - minB.z) / (maxB.z - minB.z);

    // Hardware-accelerated trilinear interpolation
    return tex3D<float>(col.sdfTexture, u, v, w);
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

__device__ bool CheckSphereBox(Collider& sphere, Collider& box, float3& normal, float& depth) {
    float3 localP = sphere.position - box.position;

    float3 clamped = make_float3(
        fmaxf(-box.dims.x, fminf(localP.x, box.dims.x)),
        fmaxf(-box.dims.y, fminf(localP.y, box.dims.y)),
        fmaxf(-box.dims.z, fminf(localP.z, box.dims.z))
    );

    float3 delta = localP - clamped;
    float dist = length(delta);

    if (dist > 1e-6f) {
        if (dist >= sphere.dims.x) return false;

        normal = delta / dist;
        depth = sphere.dims.x - dist;
        return true;
    }

    float3 overlap = make_float3(
        box.dims.x - fabsf(localP.x),
        box.dims.y - fabsf(localP.y),
        box.dims.z - fabsf(localP.z)
    );

    float3 n;
    float minOverlap;

    if (overlap.x < overlap.y && overlap.y < overlap.z) {
        minOverlap = overlap.x;
        n = make_float3(localP.x < 0.f ? -1.f : 1.f, 0.f, 0.f);
    } else if (overlap.y < overlap.z) {
        minOverlap = overlap.y;
        n = make_float3(0.f, localP.y < 0.f ? -1.f : 1.f, 0.f);
    } else {
        minOverlap = overlap.z;
        n = make_float3(0.f, 0.f, localP.z < 0.f ? -1.f : 1.f);
    }

    normal = n;
    depth = minOverlap + sphere.dims.x;
    return true;
}

__device__ bool CheckBoxBox(Collider& a, Collider& b, float3& normal, float& depth) {
    float3 delta = a.position - b.position;

    float overlapX = (a.dims.x + b.dims.x) - fabsf(delta.x);
    float overlapY = (a.dims.y + b.dims.y) - fabsf(delta.y);
    float overlapZ = (a.dims.z + b.dims.z) - fabsf(delta.z);

    if (overlapX <= 0.f || overlapY <= 0.f || overlapZ <= 0.f) {
        return false;
    }

    if (overlapX < overlapY && overlapX < overlapZ) {
        depth = overlapX;
        normal = make_float3(delta.x < 0.f ? -1.f : 1.f, 0.f, 0.f);
    } else if (overlapY < overlapZ) {
        depth = overlapY;
        normal = make_float3(0.f, delta.y < 0.f ? -1.f : 1.f, 0.f);
    } else {
        depth = overlapZ;
        normal = make_float3(0.f, 0.f, delta.z < 0.f ? -1.f : 1.f);
    }

    return true;
}

__device__ float3 CalculateColliderNormal(float3 p, Collider c) {
    float e = 0.001f;
    float3 n = make_float3(0, 0, 0);

    float3 localP = p - c.position;

    if (c.type == TYPE_SPHERE) {
        float len = length(localP);
        if (len < 1e-6f) return make_float3(0, 1, 0);
        return localP / length(localP);
    }
    if (c.type == TYPE_BOX) {
        float d = sdfBox(localP, c.dims);
        float x = sdfBox(make_float3(localP.x + e, localP.y, localP.z), c.dims) - d;
        float y = sdfBox(make_float3(localP.x, localP.y + e, localP.z), c.dims) - d;
        float z = sdfBox(make_float3(localP.x, localP.y, localP.z + e), c.dims) - d;
        n = make_float3(x, y, z);
    }
    if (c.type == TYPE_MESH) {
        float d = sdfMesh(localP, c);
        float x = sdfMesh(make_float3(localP.x + e, localP.y, localP.z), c) - d;
        float y = sdfMesh(make_float3(localP.x, localP.y + e, localP.z), c) - d;
        float z = sdfMesh(make_float3(localP.x, localP.y, localP.z + e), c) - d;
        n = make_float3(x, y, z);
    }

    float len = length(n);
    if (len < 1e-6f) return make_float3(0, 1, 0);
    return n / length(n);
}

__device__ float3 CalculateBoundsNormal(float3 p, float3 halfSize)
{
    const float e = 0.001f;
    float d = -sdfBox(p, halfSize);
    float x = (-sdfBox(make_float3(p.x + e, p.y, p.z), halfSize)) - d;
    float y = (-sdfBox(make_float3(p.x, p.y + e, p.z), halfSize)) - d;
    float z = (-sdfBox(make_float3(p.x, p.y, p.z + e), halfSize)) - d;
    float3 n = make_float3(x, y, z);
    float len = length(n);
    return len > 1e-6f ? n / len : make_float3(0, 1, 0);
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
    const int BIAS = 512;
    uint32_t ux = (uint32_t)(cellX + BIAS);
    uint32_t uy = (uint32_t)(cellY + BIAS);
    uint32_t uz = (uint32_t)(cellZ + BIAS);
    ux &= 0x3FFu; uy &= 0x3FFu; uz &= 0x3FFu;
    return expandBits(ux) | (expandBits(uy) << 1) | (expandBits(uz) << 2);
}

__device__ uint32_t GetKeyFromHash(uint32_t hash, uint32_t hashTableSize) {
    return hash % hashTableSize;
    // return hash & (hashTableSize - 1);
}

__device__ float2 ConvertDensityToPressure(float density, float nearDensity, float targetDensity, float pressureMultiplier, float nearPressureMultiplier) {
    float densityError = density  - targetDensity;
    float pressure = densityError  * pressureMultiplier;
    pressure = max(pressure, -150.0f);
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
    // float v = radius - dst;
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

__device__ float clampf(float val, float lo, float hi) {
    return fmaxf(lo, fminf(hi, val));
}

__device__ float3 cross_f3(float3 a, float3 b)
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ float3 normalize_f3(float3 v)
{
    float len = length(v);
    return len > 0.000001f ? v / len : make_float3(0.0f, 1.0f, 0.0f);
}

__global__
// __launch_bounds__(256, 2)
void ApplyPressureForces_Optimized(
    const float* __restrict__ predX,
    const float* __restrict__ predY,
    const float* __restrict__ predZ,
    float* velX, float* velY, float* velZ,
    const float* __restrict__ densities,
    const float* __restrict__ nearDensities,
    float* trappedAir,
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

    float weightedVelocityDifference = 0.0f;
    // trappedAir[i] = 0.0f;

    if (i < numParticles) {
        samplePos     = make_float3(__ldg(&predX[i]),
                                    __ldg(&predY[i]),
                                    __ldg(&predZ[i]));
        myVel         = make_float3(__ldg(&velX[i]),
                                    __ldg(&velY[i]),
                                    __ldg(&velZ[i]));
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

    if (i >= numParticles) return;

    float3 pressureForce  = make_float3(0.f, 0.f, 0.f);
    float3 viscosityForce = make_float3(0.f, 0.f, 0.f);
    int neighborCount = 0;
    const float sqrRadius = smoothingRadius * smoothingRadius;


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

            if (sqrDst <= sqrRadius && sqrDst > 1e-6f) {
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

                // Trapped air calculation
                float3 relativeVelocity = myVel - otherVel;
                float relVelMag = length(relativeVelocity);
                float3 relVelDir = relativeVelocity / max(0.000001f, relVelMag);
                float convergeWeight = (1.0f - dot(relVelDir, -dir)) * 0.5f;
                float t_influence = 1.0f - min(1.0f, dst / smoothingRadius);
                weightedVelocityDifference += relVelMag * convergeWeight * t_influence;
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

                float3 otherPos = make_float3(__ldg(&predX[j]),
                                              __ldg(&predY[j]),
                                              __ldg(&predZ[j]));
                float3 offset   = otherPos - samplePos;
                float  sqrDst   = lengthSqr(offset);

                if (sqrDst <= sqrRadius && sqrDst > 1e-6f) {
                    neighborCount++;
                    float invDst = rsqrtf(sqrDst);
                    float dst    = sqrDst * invDst;
                    float3 dir   = offset * invDst;

                    float otherDensity     = __ldg(&densities[j]);
                    float otherNearDensity = __ldg(&nearDensities[j]);
                    float3 otherVel        = make_float3(__ldg(&velX[j]),
                                                         __ldg(&velY[j]),
                                                         __ldg(&velZ[j]));

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

                    // Trapped air calculation
                    float3 relativeVelocity = myVel - otherVel;
                    float relVelMag = length(relativeVelocity);
                    float3 relVelDir = relativeVelocity / max(0.000001f, relVelMag);
                    float convergeWeight = (1.0f - dot(relVelDir, -dir)) * 0.5f;
                    float t_influence = 1.0f - min(1.0f, dst / smoothingRadius);
                    weightedVelocityDifference += relVelMag * convergeWeight * t_influence;
                }
            }
        }
    }

    float3 totalForce   = pressureForce + (viscosityForce * viscosityStrength);
    float3 acceleration = totalForce / max(myDensity, 0.0001f);
    float3 finalVel = myVel + (acceleration * dt);

    if (neighborCount < 8)
    {
        finalVel -= finalVel * (dt * 0.75f);
    }

    velX[i] = finalVel.x;
    velY[i] = finalVel.y;
    velZ[i] = finalVel.z;
    trappedAir[i] = weightedVelocityDifference;
}

__global__ void UpdatePositions(
    float* posX, float* posY, float* posZ,
    float* velX, float* velY, float* velZ,
    int numParticles, float particleSize, float boundsX, float boundsY, float boundsZ,
    float collisionDamping, float gravity, float dt,
    Collider* colliders, int numColliders, float smoothingRadius, float colliderDragModifier) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    float3 posLocal = make_float3(posX[i], posY[i], posZ[i]);
    float3 velLocal = make_float3(velX[i], velY[i], velZ[i]);

    posLocal += velLocal * dt;

    for (int k = 0; k < numColliders; k++) {
        Collider col = colliders[k];
        float3 relPos = posLocal - col.position;
        float dist = 0.0f;

        if (col.type == TYPE_SPHERE) {
            dist = sdfSphere(relPos, col.dims.x);
        } else if (col.type == TYPE_BOX) {
            dist = sdfBox(relPos, col.dims);
        } else if (col.type == TYPE_MESH) {
            dist = sdfMesh(relPos, col);
        }

        // Fluid-Structure Coupling (Drag + Buoyancy)
        if (col.isDynamic && dist < smoothingRadius) {
            float3 relativeVel = velLocal - col.velocity;
            float weight = 1.0f - (max(0.0f, dist) / smoothingRadius);

            float3 dragForce = relativeVel * weight * colliderDragModifier;

            atomicAddFloat3(&colliders[k].forceAccumulator, dragForce);

            velLocal -= dragForce * dt;

            float particleMass = 1.0f;
            float3 buoyancyDirection = make_float3(0, 1, 0);
            float buoyancyStrength = weight * particleMass * gravity;

            atomicAddFloat3(&colliders[k].forceAccumulator, buoyancyDirection * buoyancyStrength);
        }

        // Hard Collision Resolution
        if (dist < particleSize) {
            float3 normal = CalculateColliderNormal(posLocal, col);
            float penetration = particleSize - dist;

            posLocal += normal * penetration;

            float3 relativeVelocity = velLocal - col.velocity;
            float normalVel = dot(relativeVelocity, normal);

            if (normalVel < 0) {
                float3 velocityChange = normal * normalVel * (1.0f + collisionDamping);
                velLocal -= velocityChange;

                if (col.isDynamic) {
                    float particleMass = 3.0f;
                    float3 impulse = velocityChange * particleMass;
                    atomicAddFloat3(&colliders[k].forceAccumulator, impulse / dt);

                    float stiffness = 800.0f;
                    float3 reactionForce = normal * penetration * stiffness;
                    atomicAddFloat3(&colliders[k].forceAccumulator, -1.0f * reactionForce);
                }
            }
        }
    }

    const float3 halfSize = make_float3(boundsX / 2.0f, boundsY / 2.0f, boundsZ / 2.0f);
    float containerDist = -sdfBox(posLocal, halfSize);

    if (containerDist < particleSize)
    {
        float3 normal = CalculateBoundsNormal(posLocal, halfSize);
        float penetration = particleSize - containerDist;

        posLocal += normal * penetration;

        float normalVel = dot(velLocal, normal);
        if (normalVel < 0)
        {
            float3 velocityChange = normal * normalVel * (1.0f + collisionDamping);
            velLocal -= velocityChange;
        }
    }

    // const float3 edgeDst = make_float3(
    //     halfSize.x - abs(posLocal.x) - particleSize,
    //     halfSize.y - abs(posLocal.y) - particleSize,
    //     halfSize.z - abs(posLocal.z) - particleSize
    // );
    //
    // if (edgeDst.x <= 0) {
    //     posLocal.x = (halfSize.x - particleSize) * sign(posLocal.x);
    //     if (posLocal.x * velLocal.x > 0.0f) velLocal.x *= -1.0f * collisionDamping;
    // }
    // if (edgeDst.y <= 0) {
    //     posLocal.y = (halfSize.y - particleSize) * sign(posLocal.y);
    //     if (posLocal.y * velLocal.y > 0.0f) velLocal.y *= -1.0f * collisionDamping;
    // }
    // if (edgeDst.z <= 0) {
    //     posLocal.z = (halfSize.z - particleSize) * sign(posLocal.z);
    //     if (posLocal.z * velLocal.z > 0.0f) velLocal.z *= -1.0f * collisionDamping;
    // }

    const float MAX_SPEED = 60.0f;
    float speed = length(velLocal);
    if (speed > MAX_SPEED) {
        velLocal = (velLocal / speed) * MAX_SPEED;
    }

    posX[i] = posLocal.x;
    posY[i] = posLocal.y;
    posZ[i] = posLocal.z;

    velX[i] = velLocal.x;
    velY[i] = velLocal.y;
    velZ[i] = velLocal.z;
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
    } else if (col.type == TYPE_MESH) {
        extents = ((col.gridMaxBounds - col.gridMinBounds) / 2.0f) - make_float3(1.0, 1.0, 1.0);
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
        else if (colA.type == TYPE_SPHERE && colB.type == TYPE_BOX) {
            collision = CheckSphereBox(colA, colB, normal, depth);
        }
        else if (colA.type == TYPE_BOX && colB.type == TYPE_SPHERE) {
            // Flip normal because we pass (Sphere, Box)
            collision = CheckSphereBox(colB, colA, normal, depth);
            normal = normal * -1.0f;
        }
        else if (colA.type == TYPE_BOX && colB.type == TYPE_BOX) {
            collision = CheckBoxBox(colA, colB, normal, depth);
        }

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

__global__ void PredictPositions(
    float* posX, float* posY, float* posZ,
    float* predPosX, float* predPosY, float* predPosZ,
    float* velX, float* velY, float* velZ,
    int numParticles, float gravity, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    velY[i] += -1 * gravity * dt;
    predPosX[i] = posX[i] + velX[i] * dt;
    predPosY[i] = posY[i] + velY[i] * dt;
    predPosZ[i] = posZ[i] + velZ[i] * dt;
}

__global__
__launch_bounds__(256, 4)
void UpdateDensities_Optimized(
    const float* __restrict__ posX,
    const float* __restrict__ posY,
    const float* __restrict__ posZ,
    const uint32_t* __restrict__ spatialKeys,
    const uint32_t* __restrict__ startIndices,
    int numParticles,
    uint32_t hashTableSize,
    float smoothingRadius,
    float smoothingScale,
    float nearDensityScale,
    float* densities,
    float* nearDensities
) {
    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;
    const int BS = blockDim.x;

    // 1. Setup Shared Memory (SoA layout)
    extern __shared__ float s_data[];
    float* s_posX = s_data;
    float* s_posY = s_posX + BS;
    float* s_posZ = s_posY + BS;

    float3 myPos;

    // 2. Load Data into Registers and Shared Memory
    if (i < numParticles) {
        myPos = make_float3(__ldg(&posX[i]), __ldg(&posY[i]), __ldg(&posZ[i]));
    } else {
        myPos = make_float3(0.0f, 0.0f, 0.0f);
    }

    s_posX[tid] = myPos.x;
    s_posY[tid] = myPos.y;
    s_posZ[tid] = myPos.z;
    __syncthreads();

    if (i >= numParticles) return;

    float density = 0.0f;
    float nearDensity = 0.0f;
    float sqrRadius = smoothingRadius * smoothingRadius;
    const float mass = 1.0f;

    // 3. Phase 1: Shared Memory Interaction (Same Block)
    // Iterate over all particles in the current shared memory block
    // Since data is sorted, these are highly likely to be neighbors
    #pragma unroll 4
    for (int j = 0; j < BS; j++) {
        // Note: We include self-interaction here (when j == tid), which ensures
        // the particle contributes its own mass to its density.
        if ((blockIdx.x * BS + j) >= numParticles) break;

        float3 sPos = make_float3(s_posX[j], s_posY[j], s_posZ[j]);
        float sqrDst = lengthSqr(sPos - myPos);

        if (sqrDst <= sqrRadius) {
            float dst = sqrtf(sqrDst);
            density += mass * SmoothingKernel(dst, smoothingRadius, smoothingScale);
            nearDensity += mass * NearDensityKernel(dst, smoothingRadius, nearDensityScale);
        }
    }

    // 4. Phase 2: Grid Interaction (Neighboring Cells)
    int3 center = PositionToCellCoord(myPos, smoothingRadius);
    uint32_t blockStart = blockIdx.x * BS;
    uint32_t blockEnd   = blockStart + BS;

    for (int c = 0; c < 27; c++) {
        int3 cellCoord = center + CELL_OFFSETS[c];
        uint32_t key = GetKeyFromHash(HashCell(cellCoord.x, cellCoord.y, cellCoord.z), hashTableSize);
        uint32_t startIndex = __ldg(&startIndices[key]);

        if (startIndex == 0xffffffff) continue;

        for (uint32_t j = startIndex; j < numParticles; j++) {
            if (__ldg(&spatialKeys[j]) != key) break;

            // Critical optimization: Skip particles already processed in Shared Memory loop
            if (j >= blockStart && j < blockEnd) continue;

            float3 otherPos = make_float3(
                __ldg(&posX[j]),
                __ldg(&posY[j]),
                __ldg(&posZ[j])
            );

            float sqrDst = lengthSqr(otherPos - myPos);
            if (sqrDst <= sqrRadius) {
                float dst = sqrtf(sqrDst);
                density += mass * SmoothingKernel(dst, smoothingRadius, smoothingScale);
                nearDensity += mass * NearDensityKernel(dst, smoothingRadius, nearDensityScale);
            }
        }
    }

    densities[i] = density;
    nearDensities[i] = nearDensity;
}

__global__ void UpdateSpatialHash(
    float* posX, float* posY, float* posZ,
    int numParticles,
    uint32_t hashTableSize,
    float radius,
    uint32_t* spatialIndices,
    uint32_t* spatialKeys,
    uint32_t* startIndices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    float3 pos = make_float3(posX[i], posY[i], posZ[i]);

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
    float* posX, float* posY, float* posZ,
    float* sortedPosX, float* sortedPosY, float* sortedPosZ,
    float* predX, float* predY, float* predZ,
    float* sortedPredX, float* sortedPredY, float* sortedPredZ,
    float* velX, float* velY, float* velZ,
    float* sortedVelX, float* sortedVelY, float* sortedVelZ
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    int originalIndex = spatialIndices[i];

    sortedPosX[i] = posX[originalIndex];
    sortedPosY[i] = posY[originalIndex];
    sortedPosZ[i] = posZ[originalIndex];

    sortedPredX[i] = predX[originalIndex];
    sortedPredY[i] = predY[originalIndex];
    sortedPredZ[i] = predZ[originalIndex];

    sortedVelX[i] = velX[originalIndex];
    sortedVelY[i] = velY[originalIndex];
    sortedVelZ[i] = velZ[originalIndex];
}

__global__ void ReorderVelocities(
    int numParticles,
    uint32_t* spatialIndices,
    float* sortedVelX, float* sortedVelY, float* sortedVelZ,
    float* origVelX, float* origVelY, float* origVelZ
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    int originalIndex = spatialIndices[i];

    origVelX[originalIndex] = sortedVelX[i];
    origVelY[originalIndex] = sortedVelY[i];
    origVelZ[originalIndex] = sortedVelZ[i];
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

__global__ void SpawnWhiteParticles(
    const float* __restrict__ posX,
    const float* __restrict__ posY,
    const float* __restrict__ posZ,
    const float* __restrict__ velX,
    const float* __restrict__ velY,
    const float* __restrict__ velZ,
    const float* __restrict__ trappedAir,
    WhiteParticle* whiteParticles,
    uint32_t* whiteParticleCounters,
    int numParticles,
    uint32_t maxWhiteParticleCount,
    float simTime,
    float dt,
    float smoothingRadius,
    WhiteParticleParams params
)
{
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (id_x >= numParticles) return;
    // Reset survivor counter
    if (id_x == 0) whiteParticleCounters[1] = 0;

    float3 pos = make_float3(posX[id_x], posY[id_x], posZ[id_x]);
    float3 vel = make_float3(velX[id_x], velY[id_x], velZ[id_x]);
    uint32_t rngState = randomState(pos, simTime);

    float ta = trappedAir[id_x];
    float trappedAirFactor = clampf((ta - params.trappedAirMin) / max(0.000001f,
        params.trappedAirMax - params.trappedAirMin), 0.0f, 1.0f);

    float ke = dot(vel, vel);
    float kineticEnergyFactor = clampf((ke - params.kineticEnergyMin) / max(0.000001f,
        params.kineticEnergyMax - params.kineticEnergyMin), 0.0f, 1.0f);

    float particleSpawnFactor = params.trappedAirSpawnRate * trappedAirFactor * kineticEnergyFactor * dt;

    int particleSpawnCount = (int)floorf(particleSpawnFactor);
    float fractionalSpawnRemainder = particleSpawnFactor - (float)particleSpawnCount;
    if (randomValue(rngState) < fractionalSpawnRemainder) particleSpawnCount += 1;

    if (particleSpawnCount > 0) {
        uint32_t particleIndex;
        // Atomic add for the whole batch at once
        particleIndex = atomicAdd(&whiteParticleCounters[0], (uint32_t)particleSpawnCount);
        particleSpawnCount = min(particleSpawnCount, (int)(maxWhiteParticleCount - particleIndex - 1));

        float3 cylinderBase = pos;
        float3 cylinderTop  = pos + vel * dt;  // swept position over timestep

        // Build orthonormal basis around velocity direction
        float velLen = length(vel);
        float3 velDir = velLen > 0.000001f ? vel / velLen : make_float3(0.0f, 1.0f, 0.0f);

        // CalculateOrthonormal: find a vector perpendicular to velDir
        float3 arbitrary = fabsf(velDir.x) < 0.9f ? make_float3(1.0f, 0.0f, 0.0f)
                                                   : make_float3(0.0f, 1.0f, 0.0f);
        float3 cylinderAxisA = normalize_f3(cross_f3(velDir, arbitrary));
        float3 cylinderAxisB = cross_f3(cylinderAxisA, velDir);

        float cylinderRadius = smoothingRadius;

        for (int s = 0; s < particleSpawnCount; s++) {
            float randomAngle = randomValue(rngState) * 2.0f * 3.14159265f;
            float3 offsetDir  = cosf(randomAngle) * cylinderAxisA
                              + sinf(randomAngle) * cylinderAxisB;

            // sqrt for uniform distribution over disk area
            float3 baseOffset = sqrtf(randomValue(rngState)) * cylinderRadius * offsetDir;
            float3 spawnPos   = cylinderBase + baseOffset
                              + (cylinderTop - cylinderBase) * randomValue(rngState);

            WhiteParticle newParticle;
            newParticle.position        = spawnPos;
            newParticle.velocity        = vel + baseOffset;
            newParticle.remainingLifetime = lerpf(1.0f, 5.0f, randomValue(rngState));
            newParticle.pad             = 0.0f;
            whiteParticles[particleIndex + s] = newParticle;
        }
    }
}

__global__ void UpdateWhiteParticles(
    WhiteParticle* whiteParticles,
    WhiteParticle* whiteParticlesCompacted,
    WhiteParticleParams params,
    uint32_t* whiteParticleCounters,
    uint32_t maxWhiteParticleCount,
    float deltaTime,
    float gravity,
    float boundsX, float boundsY, float boundsZ,
    // Spatial hash
    const float* __restrict__ fluidPosX,
    const float* __restrict__ fluidPosY,
    const float* __restrict__ fluidPosZ,
    const float* __restrict__ fluidVelX,
    const float* __restrict__ fluidVelY,
    const float* __restrict__ fluidVelZ,
    const uint32_t* __restrict__ spatialKeys,
    const uint32_t* __restrict__ startIndices,
    int numFluidParticles,
    uint32_t hashTableSize,
    float smoothingRadius
)
{
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (id_x >= min(maxWhiteParticleCount, whiteParticleCounters[0])) return;

    WhiteParticle particle = whiteParticles[id_x];

    // Classify foam/spray/bubble
    int neighborCount = 0;
    float3 velocitySum = make_float3(0.0f, 0.0f, 0.0f);
    float weightSum = 0.0f;
    float sqrRadius = smoothingRadius * smoothingRadius;
    int3 originCell = PositionToCellCoord(particle.position, smoothingRadius);

    for (int i = 0; i < 27; i++)
    {
        int3 neighborCell = originCell + CELL_OFFSETS[i];
        uint32_t hash = HashCell(neighborCell.x, neighborCell.y, neighborCell.z);
        uint32_t key = GetKeyFromHash(hash, hashTableSize);
        uint32_t currIndex = startIndices[key];
        if (currIndex == 0xffffffff) continue;

        while (currIndex < (uint32_t)numFluidParticles)
        {
            uint32_t neighborIndex = currIndex;
            uint32_t neighborKey = spatialKeys[neighborIndex];
            currIndex++;
            if (neighborKey != key) break;

            float3 offset = make_float3(
                __ldg(&fluidPosX[neighborIndex]) - particle.position.x,
                __ldg(&fluidPosY[neighborIndex]) - particle.position.y,
                __ldg(&fluidPosZ[neighborIndex]) - particle.position.z
            );
            float sqrDst = dot(offset, offset);
            if (sqrDst < sqrRadius)
            {
                neighborCount++;
                float weight = 1.0f - sqrtf(sqrDst) / smoothingRadius;
                velocitySum += make_float3(
                    __ldg(&fluidVelX[neighborIndex]),
                    __ldg(&fluidVelY[neighborIndex]),
                    __ldg(&fluidVelZ[neighborIndex])
                ) * weight;
                weightSum += weight;
            }
        }
    }

    bool isBubble = neighborCount >= params.bubbleThreshold;
    bool isSpray = neighborCount <= params.sprayThreshold;
    bool isFoam = !(isSpray || isBubble);

    float airDragMultiplier = 0.4f;
    float effectiveGravity = gravity;
    const float boundsCollisionDampening = 0.0f;

    if (isFoam)
    {
        particle.velocity = weightSum > 0.0f ? velocitySum / weightSum : particle.velocity;
        particle.remainingLifetime -= deltaTime;
    } else if (isBubble)
    {
        const float buoyancy = 2.0f;
        const float fluidAccelMul = 3.0f;
        float3 accelerationBuoyancy = make_float3(0.0f, -gravity, 0.0f) * (1.0f - buoyancy);
        float3 accelerationFluid = weightSum > 0.0f ? (velocitySum / weightSum - particle.velocity) * fluidAccelMul :
        make_float3(0.0f, 0.0f, 0.0f);
        particle.velocity += (accelerationBuoyancy + accelerationFluid) * deltaTime;
        particle.remainingLifetime -= deltaTime;
    } else
    {
        const float dragMultiplier = 0.35f;
        float3 drag = -particle.velocity * dragMultiplier;
        particle.velocity += (make_float3(0.0f, -gravity, 0.0f) + drag) * deltaTime;
        particle.remainingLifetime -= deltaTime;
    }

    particle.position += particle.velocity * deltaTime;

    const float3 halfSize = make_float3(boundsX / 2.0f, boundsY / 2.0f, boundsZ / 2.0f);

    if (particle.position.x < -halfSize.x) {
        particle.position.x = -halfSize.x;
        particle.velocity.x *= -1.0f * (1.0f - boundsCollisionDampening);
    }
    if (particle.position.x > halfSize.x) {
        particle.position.x = halfSize.x;
        particle.velocity.x *= -1.0f * (1.0f - boundsCollisionDampening);
    }
    if (particle.position.y < -halfSize.y) {
        particle.position.y = -halfSize.y;
        particle.velocity.y *= -1.0f * (1.0f - boundsCollisionDampening);
    }
    if (particle.position.y > halfSize.y) {
        particle.position.y = halfSize.y;
        particle.velocity.y *= -1.0f * (1.0f - boundsCollisionDampening);
    }
    if (particle.position.z < -halfSize.z) {
        particle.position.z = -halfSize.z;
        particle.velocity.z *= -1.0f * (1.0f - boundsCollisionDampening);
    }
    if (particle.position.z > halfSize.z) {
        particle.position.z = halfSize.z;
        particle.velocity.z *= -1.0f * (1.0f - boundsCollisionDampening);
    }

    whiteParticles[id_x] = particle;

    if (particle.remainingLifetime > 0)
    {
        uint32_t survivorIndex = atomicAdd(&whiteParticleCounters[1], 1u);
        whiteParticlesCompacted[survivorIndex] = particle;
    }
}

__global__ void WhiteParticlePrepareNextFrame(
    WhiteParticle* whiteParticles,
    WhiteParticle* whiteParticlesCompacted,
    uint32_t* whiteParticleCounters
)
{
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t survivorCount = whiteParticleCounters[1];

    if (id_x < survivorCount)
    {
        whiteParticles[id_x] = whiteParticlesCompacted[id_x];
    }

    if (id_x == 0)
    {
        whiteParticleCounters[0] = survivorCount;
    }
}

void SPHSolver::UpdateSpatialLookup() {
    int blockSize = 256;
    int numBlock = (m_numParticles + blockSize - 1) / blockSize;

    cudaMemsetAsync(d_spatialIndices, 0xffffffff, m_maxParticles * sizeof(uint32_t));
    cudaMemsetAsync(d_startIndices, 0xffffffff, m_hashTableSize * sizeof(uint32_t));

    UpdateSpatialHash<<<numBlock, blockSize>>>(d_predX, d_predY, d_predZ, m_numParticles, m_hashTableSize, m_params.smoothingRadius, d_spatialIndices, d_spatialKeys, d_startIndices);

    if (d_sortStorage == nullptr)
    {
        cub::DeviceRadixSort::SortPairs(nullptr, m_sortStorageBytes, d_spatialKeys, d_spatialKeysSorted,
            d_spatialIndices, d_spatialIndicesSorted, m_numParticles);
        cudaMalloc(&d_sortStorage, m_sortStorageBytes);
    }

    cub::DeviceRadixSort::SortPairs(d_sortStorage, m_sortStorageBytes, d_spatialKeys, d_spatialKeysSorted,
        d_spatialIndices, d_spatialIndicesSorted, m_numParticles);

    std::swap(d_spatialKeys, d_spatialKeysSorted);
    std::swap(d_spatialIndices, d_spatialIndicesSorted);

    UpdateStartIndices<<<numBlock, blockSize>>>(d_spatialKeys, d_startIndices, m_numParticles);
    cudaDeviceSynchronize();
}

SPHSolver::SPHSolver(int maxParticles) : m_maxParticles(maxParticles), m_numParticles(0) {
    m_hashTableSize = m_maxParticles * 2;
    cudaMalloc(&d_posX, m_maxParticles * sizeof(float));
    cudaMalloc(&d_posY, m_maxParticles * sizeof(float));
    cudaMalloc(&d_posZ, m_maxParticles * sizeof(float));

    cudaMalloc(&d_sortedPosX, m_maxParticles * sizeof(float));
    cudaMalloc(&d_sortedPosY, m_maxParticles * sizeof(float));
    cudaMalloc(&d_sortedPosZ, m_maxParticles * sizeof(float));

    cudaMalloc(&d_predX, m_maxParticles * sizeof(float));
    cudaMalloc(&d_predY, m_maxParticles * sizeof(float));
    cudaMalloc(&d_predZ, m_maxParticles * sizeof(float));

    cudaMalloc(&d_velX, m_maxParticles * sizeof(float));
    cudaMalloc(&d_velY, m_maxParticles * sizeof(float));
    cudaMalloc(&d_velZ, m_maxParticles * sizeof(float));

    cudaMalloc(&d_sortedPredX, m_maxParticles * sizeof(float));
    cudaMalloc(&d_sortedPredY, m_maxParticles * sizeof(float));
    cudaMalloc(&d_sortedPredZ, m_maxParticles * sizeof(float));

    cudaMalloc(&d_sortedVelX, m_maxParticles * sizeof(float));
    cudaMalloc(&d_sortedVelY, m_maxParticles * sizeof(float));
    cudaMalloc(&d_sortedVelZ, m_maxParticles * sizeof(float));

    cudaMalloc(&d_densities, m_maxParticles * sizeof(float));
    cudaMalloc(&d_nearDensities, m_maxParticles * sizeof(float));

    cudaMalloc(&d_spatialIndices, m_maxParticles * sizeof(uint32_t));
    cudaMalloc(&d_spatialKeys, m_maxParticles * sizeof(uint32_t));

    cudaMalloc(&d_spatialIndicesSorted, m_maxParticles * sizeof(uint32_t));
    cudaMalloc(&d_spatialKeysSorted, m_maxParticles * sizeof(uint32_t));

    cudaMalloc(&d_startIndices, m_hashTableSize * sizeof(uint32_t));
    cudaMalloc(&d_colliders, 10 * sizeof(Collider));
    cudaMalloc(&d_aos_temp, m_maxParticles * 3 * sizeof(float));

    cudaMalloc(&d_trappedAir, m_maxParticles * sizeof(float));
}

SPHSolver::~SPHSolver() {
    for (const auto& col : m_colliders) {
        if (col.type == TYPE_MESH) {
            cudaDestroyTextureObject(col.sdfTexture);
            cudaFreeArray(col.sdfArray);
        }
    }

    if (d_posX) cudaFree(d_posX);
    if (d_posY) cudaFree(d_posY);
    if (d_posZ) cudaFree(d_posZ);
    if (d_sortedPosX) cudaFree(d_sortedPosX);
    if (d_sortedPosY) cudaFree(d_sortedPosY);
    if (d_sortedPosZ) cudaFree(d_sortedPosZ);
    if (d_velX) cudaFree(d_velX);
    if (d_velY) cudaFree(d_velY);
    if (d_velZ) cudaFree(d_velZ);
    if (d_sortedPredX) cudaFree(d_sortedPredX);
    if (d_sortedPredY) cudaFree(d_sortedPredY);
    if (d_sortedPredZ) cudaFree(d_sortedPredZ);
    if (d_sortedVelX) cudaFree(d_sortedVelX);
    if (d_sortedVelY) cudaFree(d_sortedVelY);
    if (d_sortedVelZ) cudaFree(d_sortedVelZ);
    if (d_densities) cudaFree(d_densities);
    if (d_nearDensities) cudaFree(d_nearDensities);
    if (d_spatialIndices) cudaFree(d_spatialIndices);
    if (d_spatialKeys) cudaFree(d_spatialKeys);
    if (d_spatialIndicesSorted) cudaFree(d_spatialIndicesSorted);
    if (d_spatialKeysSorted) cudaFree(d_spatialKeysSorted);
    if (d_sortStorage) cudaFree(d_sortStorage);
    if (d_startIndices) cudaFree(d_startIndices);
    if (d_aos_temp) cudaFree(d_aos_temp);
    if (d_colliders) cudaFree(d_colliders);
    if (d_whiteParticles) cudaFree(d_whiteParticles);
    if (d_whiteParticlesCompact) cudaFree(d_whiteParticlesCompact);
    if (d_whiteCounters) cudaFree(d_whiteCounters);
    if (d_trappedAir) cudaFree(d_trappedAir);
    if (h_whiteParticlesPinned) cudaFreeHost(h_whiteParticlesPinned);
    if (h_whiteCountPinned)     cudaFreeHost(h_whiteCountPinned);
}

void SPHSolver::init(const std::vector<float> &positions, const std::vector<float> &velocities) {
    m_numParticles = positions.size() / 3;
    if (m_numParticles > m_maxParticles) m_numParticles = m_maxParticles;

    std::vector<float> h_posX(m_numParticles);
    std::vector<float> h_posY(m_numParticles);
    std::vector<float> h_posZ(m_numParticles);

    std::vector<float> h_velX(m_numParticles);
    std::vector<float> h_velY(m_numParticles);
    std::vector<float> h_velZ(m_numParticles);

    for (int i = 0; i < m_numParticles; ++i) {
        h_posX[i] = positions[i * 3 + 0];
        h_posY[i] = positions[i * 3 + 1];
        h_posZ[i] = positions[i * 3 + 2];

        h_velX[i] = velocities[i * 3 + 0];
        h_velY[i] = velocities[i * 3 + 1];
        h_velZ[i] = velocities[i * 3 + 2];
    }

    size_t copySize = m_numParticles * sizeof(float);
    cudaMemcpy(d_posX, h_posX.data(), copySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_posY, h_posY.data(), copySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_posZ, h_posZ.data(), copySize, cudaMemcpyHostToDevice);

    cudaMemcpy(d_velX, h_velX.data(), copySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_velY, h_velY.data(), copySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_velZ, h_velZ.data(), copySize, cudaMemcpyHostToDevice);

    m_maxWhiteParticles = m_wpParams.maxWhiteParticles;
    cudaMalloc(&d_whiteParticles, m_maxWhiteParticles * sizeof(WhiteParticle));
    cudaMalloc(&d_whiteParticlesCompact, m_maxWhiteParticles * sizeof(WhiteParticle));
    cudaMalloc(&d_whiteCounters, 2 * sizeof(uint32_t));
    cudaMemset(d_whiteParticles, 0, m_maxWhiteParticles * sizeof(WhiteParticle));
    cudaMemset(d_whiteCounters, 0, 2 * sizeof(uint32_t));
    cudaMallocHost((void**)&h_whiteParticlesPinned, m_maxWhiteParticles * sizeof(WhiteParticle));
    cudaMallocHost((void**)&h_whiteCountPinned, sizeof(uint32_t));

    float h = m_params.smoothingRadius;
    float h5 = powf(h, 5.0f);
    float h6 = powf(h, 6.0f);
    float h9 = powf(h, 9.0f);
    float pi = std::numbers::pi_v<float>;

    m_params.densityScale = 15.0f / (2.0f * pi * h5);
    m_params.pressureScale = 15.0f / (pi * h5);
    m_params.nearDensityScale = 15.0f / (pi * h6);
    m_params.nearPressureScale = 45.0f / (pi * h6);
    m_params.viscosityScale = 315.0f / (64.0f * pi * h9);
    // m_params.viscosityScale = 15.0f / (pi * h6);
}

void SPHSolver::update(float dt) {
    int blockSize = 256;
    int numBlock = (m_numParticles + blockSize - 1) / blockSize;

    // Apply gravity and predict next positions
    PredictPositions<<<numBlock, blockSize>>>(
        d_posX, d_posY, d_posZ,
        d_predX, d_predY, d_predZ,
        d_velX, d_velY, d_velZ,
        m_numParticles, m_params.gravity, dt);

    UpdateSpatialLookup();

    SortData<<<numBlock, blockSize>>>(
        m_numParticles, d_spatialIndices,
        d_posX, d_posY, d_posZ, d_sortedPosX, d_sortedPosY, d_sortedPosZ,
        d_predX, d_predY, d_predZ, d_sortedPredX, d_sortedPredY, d_sortedPredZ,
        d_velX, d_velY, d_velZ, d_sortedVelX, d_sortedVelY, d_sortedVelZ);

    std::swap(d_posX, d_sortedPosX);
    std::swap(d_posY, d_sortedPosY);
    std::swap(d_posZ, d_sortedPosZ);

    std::swap(d_predX, d_sortedPredX);
    std::swap(d_predY, d_sortedPredY);
    std::swap(d_predZ, d_sortedPredZ);

    std::swap(d_velX, d_sortedVelX);
    std::swap(d_velY, d_sortedVelY);
    std::swap(d_velZ, d_sortedVelZ);

    // Calculate and apply densities
    size_t smemDensity = blockSize * 3 * sizeof(float);
    UpdateDensities_Optimized<<<numBlock, blockSize, smemDensity>>>(
         d_predX, d_predY, d_predZ,
         d_spatialKeys, d_startIndices, m_numParticles, m_hashTableSize,
         m_params.smoothingRadius, m_params.densityScale, m_params.nearDensityScale,
         d_densities, d_nearDensities
     );

    // Calculate and apply pressure forces
    size_t smemPressure = blockSize * 16 * sizeof(float);
    ApplyPressureForces_Optimized<<<numBlock, blockSize, smemPressure>>>(
        d_predX, d_predY, d_predZ,
        d_velX, d_velY, d_velZ,
        d_densities,
        d_nearDensities,
        d_trappedAir,
        d_spatialIndices, d_spatialKeys, d_startIndices, m_numParticles, m_hashTableSize,
        m_params.smoothingRadius, m_params.targetDensity, m_params.pressureMultiplier,
        m_params.nearPressureMultiplier, m_params.viscosityStrength, dt,
        m_params.pressureScale, m_params.viscosityScale, m_params.nearPressureScale
    );

    // ReorderVelocities<<<numBlock, blockSize>>>(m_numParticles, d_spatialIndices,
    //     d_sortedVelX, d_sortedVelY, d_sortedVelZ,
    //     d_velX, d_velY, d_velZ);

    // Update positions and handle collisions
    UpdatePositions<<<numBlock, blockSize>>>(
        d_posX, d_posY, d_posZ,
        d_velX, d_velY, d_velZ,
        m_numParticles,
        m_params.particleSize, m_params.boundsX, m_params.boundsY, m_params.boundsZ,
        m_params.collisionDamping, m_params.gravity, dt, d_colliders, m_numColliders, m_params.smoothingRadius, m_params.colliderDragMultiplier);

    // White particle simulation
    int wpBlockSize = 256;

    int spawnNumBlock = (m_numParticles + wpBlockSize - 1) / wpBlockSize;
    SpawnWhiteParticles<<<spawnNumBlock, wpBlockSize>>>(d_posX, d_posY, d_posZ,
        d_velX, d_velY, d_velZ,
        d_trappedAir,
        d_whiteParticles, d_whiteCounters,
        m_numParticles, m_maxWhiteParticles, m_simTime, dt, m_params.smoothingRadius, m_wpParams);

    int wpNumBlock = (m_maxWhiteParticles + wpBlockSize - 1) / wpBlockSize;
    UpdateWhiteParticles<<<wpNumBlock, wpBlockSize>>>(d_whiteParticles,
        d_whiteParticlesCompact, m_wpParams, d_whiteCounters, m_maxWhiteParticles, dt, m_params.gravity,
        m_params.boundsX, m_params.boundsY, m_params.boundsZ,
        d_posX, d_posY, d_posZ,
        d_velX, d_velY, d_velZ,
        d_spatialKeys, d_startIndices, m_numParticles, m_hashTableSize, m_params.smoothingRadius);

    WhiteParticlePrepareNextFrame<<<wpNumBlock, wpBlockSize>>>(d_whiteParticles, d_whiteParticlesCompact, d_whiteCounters);

    m_simTime += dt;

    if (m_numColliders > 0)
    {
        IntegrateColliders<<<1, 32>>>(d_colliders, m_numColliders, m_params.boundsX, m_params.boundsY, m_params.boundsZ, m_params.gravity, dt);
        ResolveColliderCollisions<<<1, m_numColliders>>>(d_colliders, m_numColliders);
    }

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
    int blockSize = 256;
    int numBlock = (m_numParticles + blockSize - 1) / blockSize;

    soa_to_aos<<<numBlock, blockSize>>>(d_aos_temp, d_posX, d_posY, d_posZ, m_numParticles);

    cudaMemcpyAsync(outPositions, d_aos_temp, m_numParticles * 3 * sizeof(float), cudaMemcpyDeviceToHost);
}

void SPHSolver::getVelocities(float* outVelocities)
{
    int blockSize = 256;
    int numBlock = (m_numParticles + blockSize - 1) / blockSize;

    soa_to_aos<<<numBlock, blockSize>>>(d_aos_temp, d_velX, d_velY, d_velZ, m_numParticles);

    cudaMemcpyAsync(outVelocities, d_aos_temp, m_numParticles * 3 * sizeof(float), cudaMemcpyDeviceToHost);
}

void SPHSolver::setParams(const SPHParams &params) {
    m_params = params;
}


void SPHSolver::getWhiteParticles(float* outPositions, uint32_t& outCount)
{
    uint32_t count = 0;
    cudaMemcpy(&count, d_whiteCounters, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    outCount = min(count, m_maxWhiteParticles);
    if (outCount == 0) return;

    std::vector<WhiteParticle> temp(outCount);
    cudaMemcpy(temp.data(), d_whiteParticles, outCount * sizeof(WhiteParticle), cudaMemcpyDeviceToHost);

    for (uint32_t i = 0; i < outCount; ++i)
    {
        outPositions[i * 3 + 0] = temp[i].position.x;
        outPositions[i * 3 + 1] = temp[i].position.y;
        outPositions[i * 3 + 2] = temp[i].position.z;
    }
}

void SPHSolver::getWhiteParticles(std::vector<WhiteParticle>& out)
{
    uint32_t count = 0;
    cudaMemcpy(&count, d_whiteCounters, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    count = min(count, m_maxWhiteParticles);
    if (count == 0) return;
    out.resize(count);
    cudaMemcpy(out.data(), d_whiteParticles, count * sizeof(WhiteParticle), cudaMemcpyDeviceToHost);
}

void SPHSolver::requestWhiteParticles()
{
    cudaMemcpyAsync(h_whiteCountPinned, d_whiteCounters, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(h_whiteParticlesPinned, d_whiteParticles, m_maxWhiteParticles * sizeof(WhiteParticle),
        cudaMemcpyDeviceToHost);
}

void SPHSolver::finalizeWhiteParticles(std::vector<WhiteParticle>& out)
{
    uint32_t count = min(*h_whiteCountPinned, m_maxWhiteParticles);
    out.assign(h_whiteParticlesPinned, h_whiteParticlesPinned + count);
}
