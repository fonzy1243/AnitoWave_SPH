cmake_minimum_required(VERSION 4.1)
project(AnitoWave_SPH CUDA CXX)

set(CMAKE_CUDA_STANDARD 23)

if (MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif ()

include_directories(${CMAKE_SOURCE_DIR}/include)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "GLFW lib only")

add_subdirectory(.\\glfw)

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