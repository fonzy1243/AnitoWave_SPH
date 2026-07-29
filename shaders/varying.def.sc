vec3 a_position     : POSITION;
vec3 a_normal       : NORMAL;
vec4 a_color0       : COLOR0;
vec2 a_texcoord0    : TEXCOORD0;

vec4 v_color        : COLOR0;
vec4 v_color0       : COLOR1;
vec2 v_uv           : TEXCOORD0;
vec3 v_normal       : TEXCOORD1;
vec2 v_texcoord0    : TEXCOORD2;

vec4 i_data0        : TEXCOORD7;
vec4 i_data1        : TEXCOORD6;

vec3 v_posView      : TEXCOORD3;
vec2 v_localPos     : TEXCOORD4;

vec3 v_dir          : TEXCOORD5;