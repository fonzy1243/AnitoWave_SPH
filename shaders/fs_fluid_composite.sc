$input v_uv

#include <bgfx_shader.sh>

SAMPLER2D(s_depth, 0);
SAMPLER2D(s_thickness, 1);
SAMPLER2D(s_sceneColor, 2);

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
    vec4 sceneColor = texture2D(s_sceneColor, v_uv);

    // Discard background pixels
    if (centerDepth <= 0.0)
    {
        // Output background color (e.g., dark gray)
        gl_FragColor = sceneColor;
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
    float thicknessMultiplier = 0.35;
    float thickness = rawThickness * thicknessMultiplier;
    float edgeMask = smoothstep(0.0, 0.01, thickness);

    float iorAir = 1.0;
    float iorFluid = 1.333;
    vec3 refractDir = refract(worldViewDir, worldNormal, iorAir / iorFluid);

    vec3 extinctionCoefficients = vec3(2.5, 0.8, 0.35);
    float transmittance = exp(-thickness * extinctionCoefficients);

    // Refraction
    float refractionMultiplier = 0.76;
    vec2 refractOffset = refractDir.xy * thickness * refractionMultiplier;

    float maxOffset = 0.14;
    refractOffset = refractOffset * min(1.0, maxOffset);

    vec2 refractUV = clamp(v_uv + refractOffset, 0.0, 1.0);
    float dispersion = min(length(refractOffset), maxOffset) * 0.08;
    float r = texture2D(s_sceneColor, clamp(v_uv + refractOffset * (1.0 + dispersion), 0.0, 1.0)).r;
    float g = texture2D(s_sceneColor, clamp(v_uv + refractOffset,                      0.0, 1.0)).g;
    float b = texture2D(s_sceneColor, clamp(v_uv + refractOffset * (1.0 - dispersion), 0.0, 1.0)).b;
    vec3 backgroundCol = vec3(r, g, b);

    vec3 ambientScatterCol = vec3(0.04, 0.18, 0.35);
    vec3 refractCol = mix(ambientScatterCol, backgroundCol, transmittance);

    // Reflection
    vec3 reflectDir = reflect(worldViewDir, worldNormal);
    vec3 skyCol = vec3(0.6, 0.8, 1.0);
    vec3 groundCol = vec3(0.3, 0.4, 0.5);
    vec3 reflectCol = mix(groundCol, skyCol, clamp(reflectDir.y, 0.0, 1.0));

    // Fresnel
    float f0 = pow((iorAir - iorFluid) / (iorAir + iorFluid), 2.0);
    float facingRatio = max(dot(-worldViewDir, worldNormal), 0.0);
    float fresnel = f0 + (1.0 - f0) * pow(1.0 - facingRatio, 5.0);

    vec3 finalColor = mix(refractCol, reflectCol, fresnel);

    // Specular
    vec3 sunDir = normalize(vec3(0.5, 1.0, 0.5));
    vec3 halfVector = normalize(sunDir - worldViewDir);
    float specular = pow(max(dot(worldNormal, halfVector), 0.0), 200.0) * edgeMask;

    finalColor += vec3(1.0, 1.0, 1.0) * specular;

    float alpha = smoothstep(0.0, 0.1, rawThickness) * edgeMask;
    gl_FragColor = vec4(mix(sceneColor.rgb, finalColor, alpha), 1.0);
}