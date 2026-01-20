cmake_minimum_required(VERSION 4.1)
project(AnitoWave_SPH CUDA CXX)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CXX_STANDARD 23)

if (MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif ()

include_directories(${CMAKE_SOURCE_DIR}/include)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "GLFW lib only")

add_subdirectory(.\\glfw)

# -----------------
# Shader compilation
# -----------------

set(SHADER_SOURCE_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SHADER_BINARY_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_BINARY_DIR})

set(SHADERC_EXE "${CMAKE_SOURCE_DIR}/bgfx/.build/win64_vs2022/bin/shadercDebug.exe")
set(BGFX_INCLUDE_PATH "${CMAKE_SOURCE_DIR}/bgfx/src")

if(WIN32)
    set(SHADER_PLATFORM "windows")
    set(SHADER_PROFILE "spirv")
    set(SHADER_SUBDIR "spirv")
else ()
    set(SHADER_PLATFORM "linux")
    set(SHADER_PROFILE "spirv")
    set(SHADER_SUBDIR "spirv")
endif ()

set(SHADER_OUTPUT_DIR "${SHADER_BINARY_DIR}/${SHADER_SUBDIR}")
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

file(GLOB SHADER_SOURCES "${SHADER_SOURCE_DIR}/*.sc")
set(COMPILED_SHADERS)

foreach (SHADER_FILE ${SHADER_SOURCES})
    get_filename_component(FILENAME ${SHADER_FILE} NAME_WE)

    if(FILENAME MATCHES "^vs_")
        set(SHADER_TYPE "vertex")
    elseif (FILENAME MATCHES "^fs_")
        set(SHADER_TYPE "fragment")
    elseif (FILENAME MATCHES "^cs_")
        set(SHADER_TYPE "compute")
    else ()
        continue()
    endif ()

    set(OUTPUT_FILE "${SHADER_OUTPUT_DIR}/${FILENAME}.bin")

    add_custom_command(
            OUTPUT ${OUTPUT_FILE}
            COMMAND ${SHADERC_EXE}
            ARGS
                -f ${SHADER_FILE}
                -o ${OUTPUT_FILE}
                --platform ${SHADER_PLATFORM}
                --type ${SHADER_TYPE}
                --profile ${SHADER_PROFILE}
                -i ${BGFX_INCLUDE_PATH}
                --verbose
            DEPENDS ${SHADER_FILE}
            COMMENT "Compiling ${SHADER_TYPE} shader: ${FILENAME}"
    )

    list(APPEND COMPILED_SHADERS ${OUTPUT_FILE})
endforeach ()

add_custom_target(compile_shaders ALL DEPENDS ${COMPILED_SHADERS})

file(GLOB_RECURSE sources
        "${CMAKE_SOURCE_DIR}/src/*.c"
        "${CMAKE_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/*.cu"
)

add_executable(AnitoWave_SPH ${sources})

target_include_directories(AnitoWave_SPH PRIVATE
        bgfx/include
        bx/include
        bimg/include
)

target_include_directories(AnitoWave_SPH PRIVATE
        bgfx/3rdparty
        bgfx/examples/common
)

target_link_directories(AnitoWave_SPH PRIVATE ./bgfx/.build/win64_vs2022/bin/)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    if (NOT EXISTS "${PROJECT_SOURCE_DIR}/bgfx/.build/win64_vs2022/bin/bgfxDebug.lib")
        message(FATAL_ERROR "BGFX not build as debug.")
    endif ()
    target_link_libraries(AnitoWave_SPH
            bgfxDebug
            bimgDebug
            bxDebug
            bimg_decodeDebug
    )
    target_compile_definitions(AnitoWave_SPH PRIVATE BX_CONFIG_DEBUG)
elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    if (NOT EXISTS "${PROJECT_SOURCE_DIR}/bgfx/.build/win64_vs2022/bin/bgfxRelease.lib")
        message(FATAL_ERROR "BGFX not build as release.")
    endif ()
    target_link_libraries(AnitoWave_SPH
            bimg_decodeRelease
            bgfxRelease
            bimgRelease
            bxRelease
    )
endif ()

set_target_properties(AnitoWave_SPH PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(AnitoWave_SPH glfw)

add_dependencies(AnitoWave_SPH compile_shaders)