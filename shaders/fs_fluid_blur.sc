$input v_uv

#include <bgfx_shader.sh>

SAMPLER2D(s_depth, 0);

uniform vec4 u_blurParams;
uniform vec4 u_blurFalloff;

float GaussianWeight(float x, float y, float sigma)
{
    float sqrDst = x * x + y * y;
    float c = 2.0 * sigma * sigma;
    return exp(-sqrDst / c);
}

void main()
{
    float centerDepth = texture2D(s_depth, v_uv).r;

    // Don't blur the neighbors if pixel is empty space
    if (centerDepth <= 0.0)
    {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float sum = 0.0;
    float weightSum = 0.0;

    vec2 texelDelta = u_blurParams.xy;
    int blurSize = int(u_blurParams.z);
    float sigma = u_blurParams.w;
    float depthFactor = u_blurFalloff.x;

    for (int x = -blurSize; x <= blurSize; x++)
    {
        vec2 sampleUV = v_uv + (texelDelta * float(x));
        float sampleDepth = texture2D(s_depth, sampleUV).r;

        // Ignore the background clear color (0.0) so we don't bleed the background into the fluid
        if (sampleDepth > 0.0)
        {
            float depthDifference = (centerDepth - sampleDepth) * depthFactor;
            float depthWeight = exp(-(depthDifference * depthDifference));
            float gaussWeight = GaussianWeight(float(x), 0, sigma);
            float totalWeight = depthWeight * gaussWeight;
            weightSum += totalWeight;
            sum += sampleDepth * totalWeight;
        }
    }

    // Prevent division by zero if we only sampled background pixels
    if (weightSum > 0.0) {
        gl_FragColor = vec4(sum / weightSum, 0.0, 0.0, 1.0);
    } else {
        gl_FragColor = vec4(centerDepth, 0.0, 0.0, 1.0);
    }
}