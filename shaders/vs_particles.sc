$input a_position, a_color0, i_data0, i_data1
$output v_color0, v_normal

#include <bgfx_shader.sh>

uniform vec4 u_particleRadius;

void main()
{
    vec3 instancePos = i_data0.xyz;
    vec4 instanceColor = i_data1;

    vec3 worldPos = instancePos + a_position * u_particleRadius.x;

    gl_Position = mul(u_modelViewProj, vec4(worldPos, 1.0));
    v_color0 = a_color0 * instanceColor;
    v_normal = a_position;
}