$input v_localPos

#include <bgfx_shader.sh>

uniform vec4 u_particleRadius;

void main()
{
    float sqrDst = dot(v_localPos, v_localPos);
    if (sqrDst > 1.0) discard;

    // Calculate the physical thickness of the sphere at this pixel
    float zThickness = sqrt(1.0 - sqrDst);
    float trueThickness = zThickness * u_particleRadius.x;

    // We output the thickness to the red channel.
    gl_FragColor = vec4(trueThickness, 0.0, 0.0, 1.0);
}