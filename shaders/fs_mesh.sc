$input v_color0, v_normal, v_texcoord0

#include <bgfx_shader.sh>

SAMPLER2D(s_albedo, 0);

void main()
{
    vec4 albedo = texture2D(s_albedo, v_texcoord0);
    albedo *= v_color0;

    if (albedo.a < 0.1)
        discard;

    gl_FragDepth = gl_FragCoord.z - 0.00001;

    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.5));
    float nDotL = max(dot(normalize(v_normal), lightDir), 0.0);

    vec3 color = albedo.rgb * (0.3 + 0.7 * nDotL);

    gl_FragColor = vec4(color, albedo.a);
}