cmake_minimum_required(VERSION 2.8)

project(mesh)

include(CheckIncludeFile)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -pendantic -Wall -O3 -march=core2")

add_executable(mesh
        src/sim/act.c
        src/sim/bp.c
        src/sim/cli.c
        src/sim/cmd.c
        src/sim/engine.c
        src/sim/erps.c
        src/sim/error.c
        src/sim/main.c
        src/sim/math.c
        src/sim/matrix.c
        src/sim/pprint.c
        src/sim/rnn_unfold.c
        src/sim/session.c
        src/sim/set.c
        src/sim/stats.c
        src/sim/network.c
        src/sim/vector.c)

#add_executable(cl
#        src/tools/cmp_layer.c
#        src/sim/math.c
#        src/sim/matrix.c
#        src/sim/vector.c)
