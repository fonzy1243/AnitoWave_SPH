cmake_minimum_required(VERSION 4.1)
project(AnitoWave_SPH CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(AnitoWave_SPH main.cu)

set_target_properties(AnitoWave_SPH PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
