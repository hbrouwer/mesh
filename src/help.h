/*
 * Copyright 2012-2022 Harm Brouwer <me@hbrouwer.eu>
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

                /*****************
                 **** general ****
                 *****************/

#define TOPIC_ABOUT \
"         ______                                                          \n" \
"    __---   )  --_      - - - - - - - - - - - - - - - - - - - - - - - - -\n" \
"  --       /      -_                                                     \n" \
" /     o  (         )   Mesh: https://github.com/hbrouwer/mesh           \n" \
"(     o   ____  o    )  (c) 2012-2022 Harm Brouwer <me@hbrouwer.eu>      \n" \
"(    o _--     o      )                                                  \n" \
" (____/       o _____)  Licensed under the Apache License, Version 2.0   \n" \
"      (____  ---  )                                                      \n" \
"           \\ \\-__/      - - - - - - - - - - - - - - - - - - - - - - - - -\n" \
"                                                                         \n" \

/* for Matt ... */
#define TOPIC_ABOOT \
"         ______                                                          \n" \
"    __---   )  --_      - - - - - - - - - - - - - - - - - - - - - - - - -\n" \
"  --       /      -_                                                     \n" \
" /     \x1b[1m\x1b[5;33mo\x1b[0m  (         )   Mesh: https://github.com/hbrouwer/mesh           \n" \
"(     \x1b[1m\x1b[5;33mo\x1b[0m   ____  \x1b[1m\x1b[5;36mo\x1b[0m    )  (c) 2012-2022 Harm Brouwer <me@hbrouwer.eu>      \n" \
"(    \x1b[1m\x1b[5;33mo\x1b[0m _--     \x1b[1m\x1b[5;36mo\x1b[0m      )                                                  \n" \
" (____/       \x1b[1m\x1b[5;36mo\x1b[0m _____)  Licensed under the Apache License, Version 2.0   \n" \
"      (____  ---  )                                                      \n" \
"           \\ \\-__/      - - - - - - - - - - - - - - - - - - - - - - - - -\n" \
"                                                                         \n" \

#define TOPIC_ACTIVATION \
"# Activation functions                                                   \n" \
"                                                                         \n" \
"`set ActFunc <name> <func>`      Set the activation function of a group  \n" \
"                                                                         \n" \
"## Functions                                                             \n" \
"                                                                         \n" \
"* `logistic`                     Logistic (sigmoid) function             \n" \
"* `bipolar_sigmoid`              Bipolar sigmoid function                \n" \
"* `softmax`                      Softmax activation function             \n" \
"* `tanh`                         Hyperbolic tangent                      \n" \
"* `linear`                       Linear function                         \n" \
"* `relu`                         Rectified linear function               \n" \
"* `leaky_relu`                   Leaky rectified linear function         \n" \
"* `elu`                          Exponential linear function             \n" \
"                                                                         \n" \
"## Other relevant topics                                                 \n" \
"                                                                         \n" \
"* [groups]                       Creating groups                         \n" \

#define TOPIC_CLASSIFICATION \
"# Classification                                                         \n" \
"                                                                         \n" \
"`confusionMatrix`                Show confusion matrix                   \n" \
"`confusionStats`                 Show classification statistics          \n" \

#define TOPIC_ERROR \
"# Error functions                                                        \n" \
"                                                                         \n" \
"`set ErrFunc <name> <func>`      Set the error function of a group       \n" \
"                                                                         \n" \
"## Functions                                                             \n" \
"                                                                         \n" \
"* `sum_squares`                  Sum squared error                       \n" \
"* `cross_entropy`                Cross entropy error                     \n" \
"* `divergence`                   Divergence error                        \n" \
 
#define TOPIC_GROUPS \
"# Groups                                                                 \n" \
"                                                                         \n" \
"`createGroup <name> <size>`      Create group of specified size          \n" \
"`createBiasGroup <name>`         Create a bias group                     \n" \
"`removeGroup <name>`             Remove group from network               \n" \
"`groups`                         List all groups of the active network   \n" \
"`attachBias <name>`              Attach a bias unit to a group           \n" \
"`set InputGroup <name>`          Set the input group of the network      \n" \
"`set OutputGroup <name>`         Set the output group of the network     \n" \
"`set ActFunc <name> <func>`      Set the activation function of a group  \n" \
"`set ErrFunc <name> <func>`      Set the error function of a group       \n" \
"                                                                         \n" \
"`set ReLUAlpha <name> <value>`   Set alpha coeff. for Leaky ReLU and ELU \n" \
"`set LogisicFSC <name> <value>`  Set logistic flat spot correction       \n" \
"`set LogisicGain <name> <value>` Set logistic gain coefficient           \n" \
"                                                                         \n" \
"`showVector <type> <name>`       Show group vector                       \n" \
"                                 (type = `[units|error]`)                \n" \
"                                                                         \n" \
"## Other relevant topics                                                 \n" \
"                                                                         \n" \
"* [projections]                  Creating projections between group      \n" \
"* [activation]                   Activation functions                    \n" \
"* [error]                        Error functions                         \n" \

#define TOPIC_LEARNING \
"# Learning                                                               \n" \
"                                                                         \n" \
" `set LearningAlgorithm <name>`  Set specified learning algorithm        \n" \
"                                                                         \n" \
"## Learning algorithms and parameters                                    \n" \
"                                                                         \n" \
"* `bp`                           Backpropagation                         \n" \
"`set TargetRadius <value>`       Adjust target if output is within radius\n" \
"`set ZeroErrorRadius <value>`    No error if output is withing radius    \n" \
"* `bptt`                         Backpropagation Through Time (BPTT)     \n" \
"`set BackTicks <value>`          Sets number of backward time ticks      \n" \
"(see `bp`)                                                               \n" \
"                                                                         \n" \
"## Other relevant topics                                                 \n" \
"                                                                         \n" \
"* [updating]                     Weight update algorithms, parameters    \n" \

#define TOPIC_NETWORKS \
"# Networks                                                               \n" \
"                                                                         \n" \
"`createNetwork <name> <type>`    Create network of specified type        \n" \
"                                 (type = `[ffn|srn|rnn]`)                \n" \
"`removeNetwork <name>`           Remove network from session             \n" \
"`networks`                       List all active networks                \n" \
"`changeNetwork <name>`           Change active network                   \n" \
"`inspect`                        Show properties of active network       \n" \
"                                                                         \n" \
"`init`                           Initialize network                      \n" \
"`reset`                          Reset network                           \n" \
"`train`                          Train network                           \n" \
"`test`                           Test network on all items               \n" \
"`testVerbose`                    Show error for each item                \n" \
"`testItem <id>`                  Test network on specified item          \n" \
"                                 (id = `[<name>|<number>]`)              \n" \
"                                                                         \n" \
"`toggleResetContexts`            Toggle context resetting                \n" \
"`set InitContextUnits <value>`   Activation of initial context units     \n" \
"                                                                         \n" \
"## Other relevant topics                                                 \n" \
"                                                                         \n" \
"* [groups]                       Creating groups                         \n" \
"* [projections]                  Creating projections                    \n" \
"* [training]                     Training networks                       \n" \
"* [testing]                      Testing networks                        \n" \

#define TOPIC_PROJECTIONS \
"# Projections                                                            \n" \
"                                                                         \n" \
"`createProjection <from> <to>`   Create projection between groups        \n" \
"`removeProjection <from> <to>`   Remove projection between groups        \n" \
"`createElmanProjection <f> <t>`  Create Elman (copy) projection          \n" \
"`removeElmanProjection <f> <t>`  Remove Elman (copy) projection          \n" \
"`projections`                    List all projection in network          \n" \
"`freezeProjection <from> <to>`   Freeze projection weights               \n" \
"`unfreezeProjection <from> <to>` Unfreeze projection weights             \n" \
"                                                                         \n" \
"`showMatrix <type> <from> <to>`  Show projection matrix                  \n" \
"                                 (type = `[weights|gradients|dynamics]`) \n" \
"                                                                         \n" \
"## Other relevant topics                                                 \n" \
"                                                                         \n" \
"* [weights]                      Weight randomization, saving, loading   \n" \
/* TODO: tunnel projections */

#define TOPIC_RANDOMIZATION \
"# Randomization                                                          \n" \
"                                                                         \n" \
"`set RandomSeed <value>`         Set seed for randomization              \n" \
"`set RandomAlgorithm <name>`     Set randomization algorithm             \n" \
"                                                                         \n" \
"## Randomization algorithms and parameters                               \n" \
"                                                                         \n" \
"* `gaussian`                     Gaussian randomization                  \n" \
"`set RandomMu <value>`           Mean                                    \n" \
"`set RandomSigma <value>`        Standard deviation                      \n" \
"* `range`                        Uniform range randomization             \n" \
"`set RandomMin <value>`          Range lower bound                       \n" \
"`set RandomMax <value>`          Range upper bound                       \n" \
"* `nguyen_widrow`                Nguyen-Widrow randomization             \n" \
"`set RandomMin <value>`          Range lower bound                       \n" \
"`set RandomMax <value>`          Range upper bound                       \n" \
"* `fan_in`                       Fan-In randomization                    \n" \
"* `binary`                       Binary randomization                    \n" \

#define TOPIC_SESSION \
"# Session settings                                                       \n" \
"                                                                         \n" \
"`togglePrettyPrinting`           Toggle pretty vector/matrix printing    \n" \
"`set ColorScheme <scheme>`       Change to specified color scheme        \n" \
"                                                                         \n" \
"## Color schemes                                                         \n" \
"                                                                         \n" \
"* `blue_red`                     Dark blue to light red (inpired by Lens)\n" \
"* `blue_yellow`                  Light blue to yellow                    \n" \
"* `grayscale`                    The classic grayscale scheme            \n" \
"* `spacepigs`                    'Space Pigs' (tribute to Fasttracker II)\n" \
"* `moody_blues`                  For stormy Mondays                      \n" \
"* `for_john`                     The colors of vegetables                \n" \
"* `gray_orange`                  A bright gray to orange continuum       \n" \

#define TOPIC_SETS \
"# Example sets                                                           \n" \
"                                                                         \n" \
"`loadSet <name> <file>`          Load example set from specified file    \n" \
"`removeSet <name>`               Remove set from active network          \n" \
"`sets`                           List all sets in active network         \n" \
"`changeSet <name>`               Change active set                       \n" \
"`items`                          List all example items in the active set\n" \
"`showItem <id>`                  Show input-target pairs for an item     \n" \
"                                 (id = `[<name>|<number>]`)              \n" \

#define TOPIC_SIMILARITY \
"# Output-Target vector similarity                                        \n" \
"                                                                         \n" \
"`similarityMatrix`               Show output-target similarity matrix    \n" \
"`similarityStats`                Show output-target similarity statistics\n" \
"`set similarityMetric <met>`     Set vector similarity metric            \n" \
"                                                                         \n" \
"## Metrics                                                               \n" \
"                                                                         \n" \
"* `inner_product`                Inner product                           \n" \
"* `harmonic_mean`                Harmonic mean                           \n" \
"* `tanimoto`                     Tanimoto coefficient                    \n" \
"* `dice`                         Dice coefficient                        \n" \
"* `pearson_correlation`          Pearson's correlation coefficient       \n" \

#define TOPIC_TESTING \
"# Testing                                                                \n" \
"                                                                         \n" \
"`test`                           Test network on all items               \n" \
"`testItem <name>`                Test network on specified item          \n" \
"`set TargetRadius <value>`       Adjust target if output is within radius\n" \
"`set ZeroErrorRadius <value>`    No error if output is withing radius    \n" \
"`set ErrorThreshold <value>`     Error threshold to reach for each item  \n" \
"                                                                         \n" \
"`recordUnits <group> <file>`     Record unit activations to file         \n" \
"                                                                         \n" \
"## Other relevant topics                                                 \n" \
"                                                                         \n" \
"* [classification]               Confusion matrix and statistics         \n" \
"* [similarity]                   Output-Target vector similarity         \n" \

#define TOPIC_TOPICS \
"# Help topics                                                            \n" \
"                                                                         \n" \
"## General                                                               \n" \
"                                                                         \n" \
"* [about]                        Show version and copyright information  \n" \
"* [activation]                   Activation functions                    \n" \
"* [classification]               Confusion matrix and statistics         \n" \
"* [error]                        Error functions                         \n" \
"* [groups]                       Creating groups                         \n" \
"* [learning]                     Learning algorithms, parameters         \n" \
"* [networks]                     Creating different network architectures\n" \
"* [projections]                  Creating projections                    \n" \
"* [randomization]                Randomization algorithms, parameters    \n" \
"* [session]                      Session settings                        \n" \
"* [sets]                         Training and testing examples           \n" \
"* [similarity]                   Output-Target vector similarity         \n" \
"* [testing]                      Testing networks                        \n" \
"* [topics]                       This list of all available help topics  \n" \
"* [training]                     Training networks                       \n" \
"* [updating]                     Update algorithms, parameters           \n" \
"* [usage]                        Show command line usage and arguments   \n" \
"* [weights]                      Weight randomization, saving, loading   \n" \
"* [welcome]                      Show welcome message                    \n" \
"                                                                         \n" \
"## Modules                                                               \n" \
"                                                                         \n" \
"* [module_dss]                   Distributed Situation-state Space (DSS) \n" \
"* [module_erp]                   Event-Related brain Potentials (ERPs)   \n" \

#define TOPIC_TRAINING \
"# Training                                                               \n" \
"                                                                         \n" \
"`train`                          Train network                           \n" \
"                                                                         \n" \
"## Training parameters                                                   \n" \
"                                                                         \n" \
"`set BatchSize <value>`          #examples after which to update weights \n" \
"                                 (default is #items in active set)       \n" \
"`set MaxEpochs <value>`          Maximum number of training epochs       \n" \
"`set ErrorThreshold <value>`     Stop if error drops below threshold     \n" \
"`set ReportAfter <value>`        Report progress after #epochs           \n" \
"                                                                         \n" \
"## Other relevant topics                                                 \n" \
"                                                                         \n" \
"* [learning]                     Learning algorithms, parameters         \n" \
"* [updating]                     Update algorithms, parameters           \n" \

#define TOPIC_UPDATING \
"# Weight updating                                                        \n" \
"                                                                         \n" \
"`set UpdateAlgorithm <name>`     Set weight update algorithm             \n" \
"                                                                         \n" \
"## Update algorithms and parameters                                      \n" \
"                                                                         \n" \
"* `steepest`                     Steepest (gradient) descent             \n" \
"`set LearningRate <value>`       Set learning rate (LR) coefficient      \n" \
"`set LRScaleFactor <value>`      Set LR scaling factor                   \n" \
"`set LRScaleAfter <value>`       Scale LR after \%epochs                  \n" \
"`set Momentum <value>`           Set momentenum (MN) coefficient         \n" \
"`set MNScaleFactor <value>`      Set MN scaling factor                   \n" \
"`set MNScaleAfter <value>`       Scale MN after \%epochs                  \n" \
"`set WeightDecay <value>`        Set weight decay (WD) coefficient       \n" \
"`set WDScaleFactor <value>`      Set WD scaling factor                   \n" \
"`set WDScaleAfter <value>`       Scale WD after \%epochs                  \n" \
"* `bounded`                      Bounded steepest descent                \n" \
"(see `steepest`)                                                         \n" \
"* `rprop+|irprop+`               (modified) Rprop (+ weight backtracking)\n" \
"`set RpropInitUpdate <value>`    Set initial update value for Rprop      \n" \
"`set RpropEtaMinus <value>`      Set Eta- for Rprop                      \n" \
"`set RpropEtaPlus <value>`       Set Eta+ for Rprop                      \n" \
"* `rprop-|irprop-`               (modified) Rprop (- weight backtracking)\n" \
"(see `rprop+|irprop+` and `steepest`)                                    \n" \
"* `qprop`                        Quick propagation                       \n" \
"(see `steepest`)                                                         \n" \
"* `dbd`                          Delta-Bar-Delta                         \n" \
"`set DBDRateIncrement <value>`   Set Kappa for Delta-Bar-Delta           \n" \
"`set DBDRateDecrement <value>`   Set Phi for Delta-Bar-Delta             \n" \
"(see `steepest`)                                                         \n" \
"                                                                         \n" \
"## Other relevant topics                                                 \n" \
"                                                                         \n" \
"* [learning]                     Learning algorithms, parameters         \n" \

#define TOPIC_USAGE \
"Usage: mesh [file | option]                                              \n" \
"                                                                         \n" \
"[file]:                                                                  \n" \
"Mesh will load and run the specified script file.                        \n" \
"                                                                         \n" \
"[option]:                                                                \n" \
"`--help`                         Show this help message                  \n" \
"`--version`                      Show version information                \n" \
"                                                                         \n" \
"When no arguments are specified, Mesh will start in CLI mode.            \n" \

#define TOPIC_WEIGHTS \
"# Weights                                                                \n" \
"                                                                         \n" \
"weightStats                      Show weight statistics                  \n" \
"                                                                         \n" \
"`saveWeights <file>`             Save weights to specified file          \n" \
"`loadWeights <file>`             Load weights from specified file        \n" \
"                                                                         \n" \
"## Other relevant topics                                                 \n" \
"                                                                         \n" \
"* [randomization]                Randomization algorithms, parameters    \n" \

#define TOPIC_WELCOME \
"# Welcome to Mesh                                                        \n" \
"                                                                         \n" \
"Mesh is a lightweight and versatile artificial neural network simulator, \n" \
"primarily designed as a general-purpose backpropagation simulator with   \n" \
"flexibility and extensibility in mind.                                   \n" \
"                                                                         \n" \
"## Quick-start                                                           \n" \
"                                                                         \n" \
"Mesh is command driven. Type `quit` or `exit` to leave this session.     \n" \
"Type `help` to show this information, or type `help <topic>` to show     \n" \
"help on a specific topic. Type `help topics` for a full list of topics.  \n" \
"Topics to start with include:                                            \n" \
"                                                                         \n" \
"* [about]                        Show version and copyright information  \n" \
"* [networks]                     Creating different network architectures\n" \
"* [session]                      Session settings                        \n" \
"* [sets]                         Training and testing examples           \n" \
"* [training]                     Training networks                       \n" \
"* [testing]                      Testing networks                        \n" \
"                                                                         \n" \
"Type `loadFile <file>` to load and run script file.                      \n" \

                /*****************
                 **** modules ****
                 *****************/

#define TOPIC_MODULE_DSS \
"# Distributed Situation-state Space (DSS)                                \n" \
"                                                                         \n" \
"## Comprehension scores                                                  \n" \
"                                                                         \n" \
"`dssTest`                        Show comprehension(target,output) for   \n" \
"                                 each sentence in the active set         \n" \
"`dssScores <set> <sen>`          Show comprehension(event,output) at each\n" \
"                                 word, for each event in set             \n" \
"`dssInferences <set> <sen> <th>` Show each event in set that yields      \n" \
"                                 comprehension(event,output) > |th|      \n" \
"                                                                         \n" \
"## Information theory                                                    \n" \
"                                                                         \n" \
"`dssWordInfo <set> <sen>`        Show information-theoretic metrics for  \n" \
"                                 each word, given a sentence set         \n" \
"`dssWriteWordInfo <set> <fn>`    Write information-theoretic metrics for \n" \
"                                 each word of each sentence to a file    \n" \
"                                                                         \n" \
"## Further reading                                                       \n" \
"                                                                         \n" \
"Frank, S. L., Koppen, M., Noordman, L. G., & Vonk, W. (2003). Modeling   \n" \
"    knowledge-based inferences in story comprehension. Cognitive Science,\n" \
"    27(6), 875–910.                                                      \n" \
"                                                                         \n" \
"Venhuizen, N. J., Crocker, M. W., and Brouwer, H. (2019).                \n" \
"    Expectation-based Comprehension: Modeling the Interaction of World   \n" \
"    Knowledge and Linguistic Experience. Discourse Processes, 56:3,      \n" \
"    229-255.                                                             \n" \


#define TOPIC_MODULE_ERP \
"# Event-Related brain Potentials (ERPs)                                  \n" \
"                                                                         \n" \
"`erpContrast <gen> <ctl> <tgt>`  Derive ERP estimate from the specified  \n" \
"                                 generator, and contrast each word of a  \n" \
"                                 control and target sentence             \n" \
"`erpWriteValues <n4> <p6> <fn>`  Derive N400 and P600 estimates from the \n" \
"                                 specified generators, and write values  \n" \
"                                 for each word of each sentence to file  \n" \
"                                                                         \n" \
"## Further reading                                                       \n" \
"                                                                         \n" \
"Brouwer, H., Crocker, M. W., Venhuizen, N. J., and Hoeks, J. C. J.       \n" \
"    (2017). A Neurocomputational Model of the N400 and the P600 in       \n" \
"    Language Processing. Cognitive Science, 41(S6), 1318-1352.           \n" \

struct help
{
        char *help_topic;
        char *help_text;
};

const static struct help hts[] = {
                        
        {"about",               TOPIC_ABOUT},
        {"aboot",               TOPIC_ABOOT},
        {"activation",          TOPIC_ACTIVATION},
        {"classification",      TOPIC_CLASSIFICATION},
        {"error",               TOPIC_ERROR},
        {"groups",              TOPIC_GROUPS},
        {"learning",            TOPIC_LEARNING},
        {"networks",            TOPIC_NETWORKS},
        {"projections",         TOPIC_PROJECTIONS},
        {"randomization",       TOPIC_RANDOMIZATION},
        {"session",             TOPIC_SESSION},
        {"sets",                TOPIC_SETS},
        {"similarity",          TOPIC_SIMILARITY},
        {"testing",             TOPIC_TESTING},
        {"topics",              TOPIC_TOPICS},
        {"training",            TOPIC_TRAINING},
        {"updating",            TOPIC_UPDATING},
        {"usage",               TOPIC_USAGE},
        {"weights",             TOPIC_WEIGHTS},
        {"welcome",             TOPIC_WELCOME},

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {"module_dss",          TOPIC_MODULE_DSS},
        {"module_erp",          TOPIC_MODULE_ERP},

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {NULL,         NULL}
};

void help(char *help_topic);

#endif /* HELP_H */
