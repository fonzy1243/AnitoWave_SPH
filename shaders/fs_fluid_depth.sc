$input v_posView, v_localPos

#include <bgfx_shader.sh>

uniform vec4 u_particleRadius;

void main()
{
    float sqrDst = dot(v_localPos, v_localPos);
    if (sqrDst > 1.0) discard;

    float zThickness = sqrt(1.0 - sqrDst);
    float zOffset = zThickness * u_particleRadius.x;

    float linearDepth = length(v_posView) - zOffset;
    gl_FragColor = vec4(linearDepth, 0.0, 0.0, 1.0);

    float trueViewZ = v_posView.z + zOffset;
    vec4 clipPos = mul(u_proj, vec4(v_posView.xy, trueViewZ, 1.0));

    gl_FragDepth = clipPos.z / clipPos.w;
}