cmake_minimum_required(VERSION 2.8)

project(mesh)

include(CheckIncludeFile)

set(CMAKE_C_COMPILER "/opt/local/bin/gcc-mp-4.7")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -Wall -O3 -march=core2 -fopenmp")

add_executable(mesh
        src/act.c
        src/array.c
        src/bp.c
        src/classify.c
        src/cli.c
        src/cmd.c
        src/error.c
        src/main.c
        src/math.c
        src/matrix.c
        src/pprint.c
        src/random.c
        src/rnn_unfold.c
        src/sanity.c
        src/session.c
        src/set.c
        src/similarity.c
        src/stats.c
        src/test.c
        src/train.c
        src/network.c
        src/vector.c
        src/mods/erps.c)
