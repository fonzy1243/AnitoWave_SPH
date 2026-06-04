$input a_position, i_data0, i_data1
$output v_localPos

#include <bgfx_shader.sh>

uniform vec4 u_particleRadius;

void main()
{
    // Transform particle center to View Space
    vec4 viewPos = mul(u_view, vec4(i_data0.xyz, 1.0));

    // Add the quad coordinates to billboard it towards the camera
    viewPos.xy += a_position.xy * u_particleRadius.x;

    // Output strictly what the thickness fragment shader needs
    gl_Position = mul(u_proj, viewPos);
    v_localPos = a_position.xy;
}