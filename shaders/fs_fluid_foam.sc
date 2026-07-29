$input v_color, v_uv

#include <bgfx_shader.sh>

void main()
{
    vec2 coord = v_uv * 2.0 - 1.0;
    float r2 = dot(coord, coord);
    if (r2 > 1.0) discard;

    gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
}