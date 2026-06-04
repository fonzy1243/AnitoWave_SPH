$input a_position, i_data0, i_data1
$output v_posView, v_localPos

#include <bgfx_shader.sh>

uniform vec4 u_particleRadius;

void main()
{
    vec4 viewPos = mul(u_view, vec4(i_data0.xyz, 1.0));
    viewPos.xy += a_position.xy * u_particleRadius.x;

    v_posView = viewPos.xyz;
    gl_Position = mul(u_proj, viewPos);
    v_localPos = a_position.xy;
}