$input v_color, v_uv

#include <bgfx_shader.sh>

void main()
{
    vec2 coord = v_uv * 2.0 - 1.0;

    gl_FragColor = v_color;
}