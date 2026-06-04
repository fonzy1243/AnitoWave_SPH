$input v_uv

#include <bgfx_shader.sh>

SAMPLER2D(s_depth, 0);
SAMPLER2D(s_thickness, 1);

uniform vec4 u_texelSize;

vec3 ViewPos(vec2 uv)
{
    float depth = texture2D(s_depth, uv).r;

    vec4 clipSpace = vec4(uv * 2.0 - 1.0, 0.5, 1.0);
    vec4 viewPos = mul(u_invProj, clipSpace);
    vec3 viewRay = viewPos.xyz / viewPos.w;

    // Since we stored absolute linear distance, we just multiply the normalized ray
    return normalize(viewRay) * depth;
}

void main()
{
    float centerDepth = texture2D(s_depth, v_uv).r;

    // Discard background pixels
    if (centerDepth <= 0.0)
    {
        // Output background color (e.g., dark gray)
        gl_FragColor = vec4(0.2, 0.2, 0.2, 1.0);
        return;
    }

    vec3 posCenter = ViewPos(v_uv);
    // Calculate normal vector by looking at how the surface position changes
    // when taking a tiny step in both X and Y dimensions
    vec2 stepSize = u_viewTexel.xy;
    // Get depth at neighbors
    // vec3 ddx = ViewPos(v_uv + vec2(stepSize.x, 0.0)) - posCenter;
    // vec3 ddy = ViewPos(v_uv + vec2(0.0, stepSize.y)) - posCenter;
    vec3 ddx = ViewPos(v_uv + vec2(stepSize.x, 0.0)) - ViewPos(v_uv - vec2(stepSize.x, 0.0));
    vec3 ddy = ViewPos(v_uv + vec2(0.0, stepSize.y)) - ViewPos(v_uv - vec2(0.0, stepSize.y));

    // Compute view-space normal, and transform to world space
    vec3 viewNormal = normalize(cross(ddy, ddx));
    vec3 worldNormal = mul(u_invView, vec4(viewNormal, 0.0)).xyz;
    worldNormal = normalize(worldNormal);

    // ======= Lighting ========
    vec3 worldViewDir = normalize(mul(u_invView, vec4(posCenter, 0.0)).xyz);
    float rawThickness = texture2D(s_thickness, v_uv).r;
    float thicknessMultiplier = 0.05;
    float thickness = rawThickness * thicknessMultiplier;
    float edgeMask = smoothstep(0.0, 0.01, thickness);

    float iorAir = 1.0;
    float iorFluid = 1.333;
    vec3 refractDir = refract(worldViewDir, worldNormal, iorAir / iorFluid);

    vec3 extinctionCoefficients = vec3(1.5, 0.47, 0.57);

    float transmittance = exp(-thickness * extinctionCoefficients);

    float refractionMultiplier = 0.76;
    vec3 exitPos = posCenter + (refractDir * thickness * refractionMultiplier);

    vec3 floorCol = vec3(0.4, 0.4, 0.4);
    vec3 deepTrenchCol = vec3(0.05, 0.1, 0.15);
    vec3 fakeBackgroundCol = mix(deepTrenchCol, floorCol, clamp((exitPos.y + 10.0) / 20.0, 0.0, 1.0));

    vec3 ambientScatterCol = vec3(0.1, 0.3, 0.4);
    vec3 refractCol = mix(ambientScatterCol, fakeBackgroundCol, transmittance);

    // Reflection
    vec3 reflectDir = reflect(worldViewDir, worldNormal);
    vec3 skyCol = vec3(0.6, 0.8, 1.0);
    vec3 groundCol = vec3(0.3, 0.4, 0.5);
    vec3 reflectCol = mix(groundCol, skyCol, clamp(reflectDir.y, 0.0, 1.0));

    // Fresnel
    float f0 = pow((iorAir - iorFluid) / (iorAir + iorFluid), 2.0);
    float facingRatio = max(dot(-worldViewDir, worldNormal), 0.0);
    float fresnel = f0 + (1.0 - f0) * pow(1.0 - facingRatio, 5.0);
    fresnel = min(fresnel, 0.85) * edgeMask;

    vec3 finalColor = mix(refractCol, reflectCol, fresnel);

    // Specular
    vec3 sunDir = normalize(vec3(0.5, 1.0, 0.5));
    vec3 halfVector = normalize(sunDir - worldViewDir);
    float specular = pow(max(dot(worldNormal, halfVector), 0.0), 200.0) * edgeMask;

    finalColor += vec3(1.0, 1.0, 1.0) * specular;
    float alpha = smoothstep(0.0, 0.1, rawThickness);

    gl_FragColor = vec4(finalColor, alpha);
}