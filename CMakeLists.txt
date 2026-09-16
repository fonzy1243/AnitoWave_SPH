cmake_minimum_required(VERSION 4.1)
project(AnitoWave_SPH CUDA CXX)
include(FetchContent)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
endif ()

# Force MSVC / Visual Studio to automatically pass CUDA symbols to the IntelliSense engine
if(MSVC)
    add_compile_definitions($<$<BOOL:$<CONFIG>>:__CUDACC__>)
    set(CMAKE_VS_INTELISENSE_ARCHITECTURE "amd64")
endif()

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

target_compile_options(AnitoWave_SPH PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)

target_include_directories(AnitoWave_SPH PRIVATE
        ${CMAKE_BINARY_DIR}/_deps/trianglemeshdistance-src/TriangleMeshDistance/include
        $ENV{CUDA_PATH}/include
        bgfx/include
        bx/include
        bimg/include
        bgfx/3rdparty
        bgfx/examples/common
)

target_link_directories(AnitoWave_SPH PRIVATE ./bgfx/.build/win64_vs2022/bin/)

target_compile_definitions(AnitoWave_SPH PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:CCCL_IGNORE_DEPRECATED_CPP_DIALECT>
        $<$<CONFIG:Debug>:BX_CONFIG_DEBUG=1>
        $<$<CONFIG:Release>:BX_CONFIG_DEBUG=0>
)

target_link_libraries(AnitoWave_SPH
        $<$<CONFIG:Debug>:bgfxDebug>
        $<$<CONFIG:Debug>:bimgDebug>
        $<$<CONFIG:Debug>:bxDebug>
        $<$<CONFIG:Debug>:bimg_decodeDebug>
        $<$<CONFIG:Release>:bgfxRelease>
        $<$<CONFIG:Release>:bimgRelease>
        $<$<CONFIG:Release>:bxRelease>
        $<$<CONFIG:Release>:bimg_decodeRelease>
        dear-imgui
)

set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/bgfx/3rdparty/dear-imgui)
file(GLOB IMGUI_SOURCES
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
)

set(BGFX_IMGUI_DIR ${CMAKE_SOURCE_DIR}/bgfx/examples/common/imgui)
list(APPEND IMGUI_SOURCES
        ${BGFX_IMGUI_DIR}/imgui.cpp
)

add_library(dear-imgui STATIC ${IMGUI_SOURCES})
target_include_directories(dear-imgui PUBLIC
        ${IMGUI_DIR}
        ${CMAKE_SOURCE_DIR}/bgfx/include
        ${CMAKE_SOURCE_DIR}/bx/include
        ${CMAKE_SOURCE_DIR}/bimg/include
        ${CMAKE_SOURCE_DIR}/bgfx/3rdparty
)
target_link_libraries(AnitoWave_SPH dear-imgui)

target_compile_definitions(dear-imgui PRIVATE
        $<$<CONFIG:Debug>:BX_CONFIG_DEBUG=1>
        $<$<CONFIG:Release>:BX_CONFIG_DEBUG=0>
)

if (MSVC)
    target_include_directories(dear-imgui PUBLIC "${CMAKE_SOURCE_DIR}/bx/include/compat/msvc")
endif()

set_target_properties(AnitoWave_SPH PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

FetchContent_Declare(
        TriangleMeshDistance
        GIT_REPOSITORY https://github.com/InteractiveComputerGraphics/TriangleMeshDistance.git
        GIT_TAG        main
)
FetchContent_MakeAvailable(TriangleMeshDistance)

FetchContent_Declare(
        tinyobjloader
        GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
        GIT_TAG        release
)
FetchContent_MakeAvailable(tinyobjloader)

set(TINYGLTF_HEADER_ONLY ON CACHE BOOL "" FORCE)
set(TINYGLTF_INSTALL OFF CACHE BOOL "" FORCE)
set(TINYGLTF_BUILD_LOADER_EXAMPLE OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
        tinygltf
        GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
        GIT_TAG        v2.9.7
)
FetchContent_MakeAvailable(tinygltf)

target_link_libraries(AnitoWave_SPH glfw tinyobjloader tinygltf)

add_dependencies(AnitoWave_SPH compile_shaders)

# -----------------
# Shaders folder
# -----------------
add_custom_target(CopyShaders
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${SHADER_BINARY_DIR}"
        "$<TARGET_FILE_DIR:AnitoWave_SPH>/shaders"
        COMMENT "Copying compiled shaders to output directory..."
)

add_dependencies(CopyShaders compile_shaders)
add_dependencies(AnitoWave_SPH CopyShaders)

# -----------------
# Mesh folder
# -----------------
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/meshes)

add_custom_target(CopyMeshes
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/meshes"
        "$<TARGET_FILE_DIR:AnitoWave_SPH>/meshes"
        COMMENT "Copying meshes to output directory..."
)

add_dependencies(AnitoWave_SPH CopyMeshes)

# -----------------
# Textures folder
# -----------------
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/textures)

add_custom_target(CopyTextures
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/textures"
        "$<TARGET_FILE_DIR:AnitoWave_SPH>/textures"
        COMMENT "Copying textures to output directory..."
)

add_dependencies(AnitoWave_SPH CopyTextures)