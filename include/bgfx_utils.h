#ifndef ANITOWAVE_SPH_BGFX_UTILS_H
#define ANITOWAVE_SPH_BGFX_UTILS_H

#include <bx/bounds.h>
#include <bx/pixelformat.h>
#include <bx/filepath.h>
#include <bgfx/bgfx.h>

bgfx::ShaderHandle loadShader(const bx::StringView& _name);

bgfx::ProgramHandle loadProgram(const bx::StringView& _vsName, const bx::StringView& _fsName);

#endif //ANITOWAVE_SPH_BGFX_UTILS_H