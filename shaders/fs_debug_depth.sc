$input v_uv

#include <bgfx_shader.sh>

SAMPLER2D(s_depth, 0);

void main()
{
    float rawDepth = texture2D(s_depth, v_uv).r;

    float minDepth = 5.0;
    float maxDepth = 40.0;
    float gray = saturate((rawDepth - minDepth) / (maxDepth - minDepth));

    if (rawDepth <= 0.0) {
        discard;
    }

    gl_FragColor = vec4(vec3_splat(gray), 1.0);
}