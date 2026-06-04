$input a_position // Ignored
$output v_uv

#include <bgfx_shader.sh>

void main()
{
    // Generate a massive triangle covering the screen: (-1, -1), (3, -1), (-1, 3)
    float x = -1.0 + float((gl_VertexID & 1) << 2);
    float y = -1.0 + float((gl_VertexID & 2) << 1);

    v_uv.x = (x + 1.0) * 0.5;
    v_uv.y = 1.0 - ((y + 1.0) * 0.5); // Flip Y for Vulkan/bgfx UV coordinates

    gl_Position = vec4(x, y, 0.0, 1.0);
}