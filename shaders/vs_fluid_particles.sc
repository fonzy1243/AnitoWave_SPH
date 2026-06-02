$input a_position, i_data0, i_data1
$output v_color, v_uv

#include <bgfx_shader.sh>

uniform vec4 u_particleRadius;

void main()
{
    v_uv = a_position.xy * 0.5 + 0.5;

    vec3 worldCenter = i_data0.xyz;
    vec3 vel = i_data1.xyz;

    // Set color based on speed
    float speedT = clamp(length(vel) / 15, 0.0, 1.0);
    vec3 colorSlow = vec3(0.0, 0.2, 0.8);
    vec3 colorMed = vec3 (0.2, 0.9, 0.2);
    vec3 colorFast = vec3(0.9, 0.1, 0.1);

    vec3 finalColor;
    if (speedT < 0.5) {
        finalColor = mix(colorSlow, colorMed, speedT * 2.0);
    } else {
        finalColor = mix(colorMed, colorFast, (speedT - 0.5) * 2.0);
    }

    v_color = vec4(finalColor, 1.0);

    vec4 viewPos = mul(u_view, vec4(worldCenter, 1.0));
    viewPos.xy += a_position.xy * u_particleRadius.x;

    gl_Position = mul(u_proj, viewPos);
}