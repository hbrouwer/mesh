/*
 * Copyright 2012-2018 Harm Brouwer <me@hbrouwer.eu>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HELP_H
#define HELP_H

#define TOPIC_ABOUT \
"         ______                                                        \n" \
"    __---   )  --_  Mesh: http://hbrouwer.github.io/mesh/              \n" \
"  --       /      -_  Copyright 2012-2017 Harm Brouwer <me@hbrouwer.eu>\n" \
" /     o  (         )                                                  \n" \
"(     o   ____  o    )  Saarland University                            \n" \
"(    o _--     o      )  Department of Language Science and Technology \n" \
" (____/       o _____)  Psycholinguistics Group                        \n" \
"      (____  ---  )                                                    \n" \
"           \\ \\-__/  Licensed under the Apache License, Version 2.0   \n" \
"                                                                       \n" \

/* for Matt ... */
#define TOPIC_ABOOT \
"         ______                                                        \n" \
"    __---   )  --_  Mesh: http://hbrouwer.github.io/mesh/              \n" \
"  --       /      -_  Copyright 2012-2017 Harm Brouwer <me@hbrouwer.eu>\n" \
" /     \x1b[1m\x1b[5;33mo\x1b[0m  (         )                                                  \n" \
"(     \x1b[1m\x1b[5;33mo\x1b[0m   ____  \x1b[1m\x1b[5;36mo\x1b[0m    )  Saarland University                            \n" \
"(    \x1b[1m\x1b[5;33mo\x1b[0m _--     \x1b[1m\x1b[5;36mo\x1b[0m      )  Department of Language Science and Technology \n" \
" (____/       \x1b[1m\x1b[5;36mo\x1b[0m _____)  Psycholinguistics Group                        \n" \
"      (____  ---  )                                                    \n" \
"           \\ \\-__/  Licensed under the Apache License, Version 2.0   \n" \
"                                                                       \n" \

#define TOPIC_ACTIVATION \
"# Activation functions                                                 \n" \
"                                                                       \n" \
"set ActFunc <name> <func>      Set the activation function of a group  \n" \
"                                                                       \n" \
"## Functions                                                           \n" \
"                                                                       \n" \
"* [logistic]                   Logistic (sigmoid) function             \n" \
"* [bipolar_sigmoid]            A bipolar sigmoid function              \n" \
"* [softmax]                    Softmax activation function             \n" \
"* [tanh]                       Hyperbolic tangent                      \n" \
"* [linear]                     Linear function                         \n" \
"* [softplus]                   Smoothed rectified linear function      \n" \
"* [relu]                       Rectified linear function               \n" \
"* [leaky_relu]                 Leaky rectified linear function         \n" \
"* [binary_relu]                Binary rectified linear function        \n" \

#define TOPIC_ERROR \
"# Error functions                                                      \n" \
"                                                                       \n" \
"set ErrFunc <name> <func>      Set the error function of a group       \n" \
"                                                                       \n" \
"## Functions                                                           \n" \
"                                                                       \n" \
"* [sum_squares]                Sum squared error                       \n" \
"* [cross_entropy]              Cross entropy error                     \n" \
"* [divergence]                 Divergence error                        \n" \

#define TOPIC_GENERAL \
"# Welcome to Mesh                                                      \n" \
"                                                                       \n" \
"Mesh is an artificial neural network simulator, primarily designed as  \n" \
"a fast, general-purpose backpropagation simulator with flexibility and \n" \
"extensibility in mind.                                                 \n" \
"                                                                       \n" \
"## Quick-start                                                         \n" \
"                                                                       \n" \
"Mesh is command driven. Type `quit` or `exit` to leave this session.   \n" \
"Type `help` to show this information, or type `help <topic>` to show   \n" \
"help on a specific topic. Topics include:                              \n" \
"                                                                       \n" \
"* [about]                      Show version and copyright information  \n" \
"* [networks]                   Creating different network architectures\n" \
"* [sets]                       Training and testing examples           \n" \
"* [training]                   Training networks                       \n" \
"* [testing]                    Testing networks                        \n" \
"                                                                       \n" \
"Type `loadFile <file>` to load and run script file.                    \n" \
 
#define TOPIC_GROUPS \
"# Groups                                                               \n" \
"                                                                       \n" \
"createGroup <name> <size>      Create group of specified size          \n" \
"removeGroup <name>             Remove group from network               \n" \
"listGroups                     List all groups of the active network   \n" \
"attachBias <name>              Attach a bias unit to a group           \n" \
"set InputGroup <name>          Set the input group of the network      \n" \
"set OutputGroup <name>         Set the output group of the network     \n" \
"set ActFunc <name> <func>      Set the activation function of a group  \n" \
"set ErrFunc <name> <func>      Set the error function of a group       \n" \
"                                                                       \n" \
"## Other relevant topics                                               \n" \
"                                                                       \n" \
"* [projections]                Creating projections between group      \n" \
"* [activation]                 Activation functions                    \n" \
"* [error]                      Error functions                         \n" \

#define TOPIC_LEARNING \
"# Learning                                                             \n" \
"                                                                       \n" \
" set LearningAlgorithm <name>  Set specified learning algorithm        \n" \
"                                                                       \n" \
"## Learning algorithms and parameters                                  \n" \
"                                                                       \n" \
"* [bp]                         Backpropagation                         \n" \
"    set TargetRadius <val>     Adjust target if output is within radius\n" \
"    set ZeroErrorRadius <val>  No error if output is withing radius    \n" \
"* [bptt]                       Backpropagation Through Time (BPTT)     \n" \
"    set BackTicks <value>      Sets number of backward time ticks      \n" \
"    (also see `bp`)                                                    \n" \
"                                                                       \n" \
"## Other relevant topics                                               \n" \
"                                                                       \n" \
"* [update]                     Weight update algorithms, parameters    \n" \

#define TOPIC_NETWORKS \
"# Networks                                                             \n" \
"                                                                       \n" \
"createNetwork <name> <type>    Create network of specified type        \n" \
"                               (type = [ffn|srn|rnn])                  \n" \
"removeNetwork <name>           Remove network from session             \n" \
"listNetworks                   List all active networks                \n" \
"changeNetwork <name>           Change active network                   \n" \
"                                                                       \n" \
"init                           Initialize network                      \n" \
"reset                          Reset network                           \n" \
"train                          Train network                           \n" \
"test                           Test network on all items               \n" \
"testItem <name>                Test network on specified item          \n" \
"                                                                       \n" \
"## Other relevant topics                                               \n" \
"                                                                       \n" \
"* [groups]                     Creating groups                         \n" \
"* [projections]                Creating projections                    \n" \
"* [training]                   Training networks                       \n" \
"* [testing]                    Testing networks                        \n" \

#define TOPIC_PROJECTIONS \
"# Projections                                                          \n" \
"                                                                       \n" \
"createProjection <from> <to>   Create projection between groups        \n" \
"removeProjection <from> <to>   Remove projection between groups        \n" \
"createElmanProjection <from> <to>                                      \n" \
"                          ---- Create Elman (copy) projection          \n" \
"createElmanProjection <from> <to>                                      \n" \
"                          ---- Remove Elman (copy) projection          \n" \
"listProjections                List all projection in network          \n" \
"freezeProjection <from> <to>   Freeze projection weights               \n" \
"unfreezeProjection <from> <to> Unfreeze projection weights             \n" \
"                                                                       \n" \
"## Other relevant topics                                               \n" \
"                                                                       \n" \
"* [weights]                    Weight randomization, saving, loading   \n" \
/* TODO: tunnel projections */

#define TOPIC_RANDOMIZATION \
"# Randomization                                                        \n" \
"                                                                       \n" \
"set RandomSeed <val>           Set seed for randomization              \n" \
"set RandomAlgorithm <name>     Set randomization algorithm             \n" \
"                                                                       \n" \
"## Randomization algorithms and parameters                             \n" \
"                                                                       \n" \
"* [gaussian]                   Gaussian randomization                  \n" \
"    set RandomMu <val>         Mean                                    \n" \
"    set RandomSigma <val       Standard deviation                      \n" \
"* [range]                      Uniform range randomization             \n" \
"    set RandomMin <val>        Range lower bound                       \n" \
"    set RandomMax <val>        Range upper bound                       \n" \
"* [nguyen_widrow]              Nguyen-Widrow randomization             \n" \
"    set RandomMin <val>        Range lower bound                       \n" \
"    set RandomMax <val>        Range upper bound                       \n" \
"* [fan_in]                     Fan-In randomization                    \n" \
"* [binary]                     Binary randomization                    \n" \

#define TOPIC_SETS \
"# Example sets                                                         \n" \
"                                                                       \n" \
"loadSet <name> <file>          Load example set from specified file    \n" \
"removeSet <name>               Remove set from active network          \n" \
"listSets                       List all sets in active network         \n" \
"changeSet <name>               Change active set                       \n" \
"listItems                      List all example items in the active set\n" \
"showItem <name>                Show input-target pairs for an item     \n" \

#define TOPIC_TESTING \
"# Testing                                                              \n" \
"                                                                       \n" \
"test                           Test network on all items               \n" \
"testItem <name>                Test network on specified item          \n" \

#define TOPIC_TRAINING \
"# Training                                                             \n" \
"                                                                       \n" \
"train                          Train network                           \n" \
"                                                                       \n" \
"## Training parameters                                                 \n" \
"                                                                       \n" \
"set BatchSize <val>            #examples after which to update weights \n" \
"                               (default is #items in active set)       \n" \
"set MaxEpochs <val>            Maximum number of training epochs       \n" \
"set ErrorThreshold <val>       Stop if error drops below threshold     \n"\
"set ReportAfter <val>          Report progress after #epochs           \n" \
"                                                                       \n" \
"## Other relevant topics                                               \n" \
"                                                                       \n" \
"* [learning]                   Learning algorithms, parameters         \n" \
"* [update]                     Update algorithms, parameters           \n" \

#define TOPIC_UPDATE \
"# Weight updating                                                      \n" \
"                                                                       \n" \
"set UpdateAlgorithm <name>     Set weight update algorithm             \n" \
"                                                                       \n" \
"## Update algorithms and parameters                                    \n" \
"                                                                       \n" \
"* [steepest]                   Steepest (gradient) descent             \n" \
"    set LearningRate <val>     Set learning rate (LR) coefficient      \n" \
"    set LRScaleFactor <val>    Set LR scaling factor                   \n" \
"    set LRScaleAfter <val>     Scale LR after \%epochs                 \n" \
"    set Momentum <val>         Set momentenum (MN) coefficient         \n" \
"    set MNScaleFactor <val>    Set MN scaling factor                   \n" \
"    set MNScaleAfter <val>     Scale MN after \%epochs                 \n" \
"    set WeightDecay <val>      Set weight decay (WD) coefficient       \n" \
"    set WDScaleFactor <val>    Set WD scaling factor                   \n" \
"    set WDScaleAfter <val>     Scale WD after \%epochs                 \n" \
"* [bounded]                    Bounded steepest descent                \n" \
"    (see `steepest`)                                                   \n" \
"* [rprop+|irprop+]             (modified) Rprop (+ weight backtracking)\n" \
"    set RpropInitUpdate <val>  Set initial update value for Rprop      \n" \
"    set RpropEtaMinus <val>    Set Eta- for Rprop                      \n" \
"    set RpropEtaPlus <val>     Set Eta+ for Rprop                      \n" \
"* [rprop-|irprop-]             (modified) Rprop (- weight backtracking)\n" \
"    (see `rprop+|irprop+` and `steepest`)                              \n" \
"* [qprop]                      Quick propagation                       \n" \
"    (see `steepest`)                                                   \n" \
"* [dbd]                        Delta-Bar-Delta                         \n" \
"    set DBDRateIncrement <val> Set Kappa for Delta-Bar-Delta           \n" \
"    set DBDRateDecrement <val> Set Phi for Delta-Bar-Delta             \n" \
"    (also see `steepest`)                                              \n" \
"                                                                       \n" \
"## Other relevant topics                                               \n" \
"                                                                       \n" \
"* [learning]                   Learning algorithms, parameters         \n" \

#define TOPIC_USAGE \
"Usage: mesh <file> [options]                                           \n" \
"                                                                       \n" \
"    --help                     Show this help message                  \n" \

#define TOPIC_WEIGHTS \
"# Weights                                                              \n" \
"                                                                       \n" \
"weightStats                    Show weight statistics                  \n" \
"                                                                       \n" \
"saveWeights <file>             Save weights to specified file          \n" \
"loadWeights <file>             Load weights from specified file        \n" \
"                                                                       \n" \
"## Other relevant topics                                               \n" \
"                                                                       \n" \
"* [randomization]              Randomization algorithms, parameters    \n" \


struct help
{
        char *help_topic;
        char *help_text;
};

const static struct help hts[] = {
        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {"about",               TOPIC_ABOUT},
        {"aboot",               TOPIC_ABOOT},
        {"activation",          TOPIC_ACTIVATION},
        {"error",               TOPIC_ERROR},
        {"general",             TOPIC_GENERAL},
        {"groups",              TOPIC_GROUPS},
        {"learning",            TOPIC_LEARNING},
        {"networks",            TOPIC_NETWORKS},
        {"projections",         TOPIC_PROJECTIONS},
        {"randomization",       TOPIC_RANDOMIZATION},
        {"sets",                TOPIC_SETS},
        {"testing",             TOPIC_TESTING},
        {"training",            TOPIC_TRAINING},
        {"update",              TOPIC_UPDATE},
        {"usage",               TOPIC_USAGE},
        {"weights",             TOPIC_WEIGHTS},

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {NULL,         NULL}
};

void help(char *help_topic);

#endif /* HELP_H */
