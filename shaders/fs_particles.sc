$input v_color0, v_normal

#include <bgfx_shader.sh>

void main()
{
    vec3 lightDir = normalize(vec3(0.5, 1.0, -0.5));
    vec3 normal = normalize(v_normal);
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 ambient = vec3_splat(0.3);
    vec3 diffuse = vec3_splat(0.7) * diff;
    vec3 finalColor = v_color0.rgb * (ambient + diffuse);

    gl_FragColor = vec4(finalColor, v_color0.a);
}