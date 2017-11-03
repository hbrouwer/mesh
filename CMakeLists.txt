cmake_minimum_required(VERSION 2.8)

project(mesh)

include(CheckIncludeFile)

IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -Wall -O3 -march=core2")# -DFAST_EXP")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        # Use GNU and POSIX source directives
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -std=c99 -Wall -O3 -march=core2 -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L")# -DFAST_EXP")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

# Sources
set(SOURCE_FILES 
        src/act.c
        src/array.c
        src/bp.c
        src/classify.c
        src/cli.c
        src/cmd.c
        src/error.c
        src/help.c
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
        src/modules/dss.c
        src/modules/dynsys.c
        src/modules/erp.c)

# mesh
add_executable(mesh ${SOURCE_FILES})
target_link_libraries(mesh m)

# mesh + openmp
#add_executable(mesh-mp ${SOURCE_FILES})
#target_link_libraries(mesh-mp m)
#set_target_properties(mesh-mp PROPERTIES COMPILE_FLAGS -fopenmp LINK_FLAGS -fopenmp)
