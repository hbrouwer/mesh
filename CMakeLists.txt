##
# Copyright 2012-2020 Harm Brouwer <me@hbrouwer.eu>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##

cmake_minimum_required(VERSION 3.4)

project(Mesh)

option(FAST_EXP "Enable fast exponentional"     OFF)
option(OPENMP   "Enable OpenMP support"         ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -Wall -O3 -march=core2 -D_POSIX_C_SOURCE=200809L")

set(Mesh_SOURCE_FILES
        src/act.c src/array.c src/bp.c src/classify.c src/cli.c src/cmd.c
        src/engine.c src/error.c src/help.c src/main.c src/math.c src/matrix.c
        src/network.c src/pprint.c src/random.c src/record.c src/rnn_unfold.c
        src/session.c src/set.c src/similarity.c src/stats.c src/test.c
        src/train.c src/vector.c src/verify.c src/modules/dss.c
        src/modules/erp.c)

add_executable(mesh ${Mesh_SOURCE_FILES})
target_link_libraries(mesh m)

#### fast exponentional ####

if(FAST_EXP)
        add_definitions(-DFAST_EXP)
endif(FAST_EXP)

#### OpenMP ####

if(OPENMP)
        set_target_properties(
                mesh PROPERTIES
                COMPILE_FLAGS -fopenmp
                LINK_FLAGS    -fopenmp
        )
endif(OPENMP)
