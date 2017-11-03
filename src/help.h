/*
 * Copyright 2012-2017 Harm Brouwer <me@hbrouwer.eu>
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

#define TOPIC_USAGE \
"Usage:                                                                 \n" \
"                                                                       \n" \
"       --help: show this help message;                                 \n" 

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
"* networks: creating different network architectures and topologies;   \n" \
"* training: training networks;                                         \n"
 
#define TOPIC_NETWORKS \
"# Networks                                                             \n"

#define TOPIC_TRAINING \
"# Training                                                             \n"

struct help
{
        char *help_topic;
        char *help_text;
};

const static struct help hts[] = {
        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {"usage",               TOPIC_USAGE},
        {"general",             TOPIC_GENERAL},
        {"networks",            TOPIC_NETWORKS},
        {"training",            TOPIC_TRAINING},

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {NULL,         NULL}
};

void help(char *help_topic);

#endif /* HELP_H */
