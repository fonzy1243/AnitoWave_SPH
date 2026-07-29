$input a_position, i_data0, i_data1
$output v_color, v_uv

#include <bgfx_shader.sh>

uniform vec4 u_particleRadius;

void main()
{
    v_uv = a_position.xy * 0.5 + 0.5;

    vec3 worldCenter = i_data0.xyz;
    float lifetime = i_data1.w;
    float scale = clamp(lifetime / 2.5, 0.0, 1.0);

    v_color = vec4(1.0, 1.0, 1.0, 1.0);

    vec4 viewPos = mul(u_view, vec4(worldCenter, 1.0));
    viewPos.xy += a_position.xy * u_particleRadius.x * scale;

    gl_Position = mul(u_proj, viewPos);
}