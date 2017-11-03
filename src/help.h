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

#define TOPIC_GENERAL \
"### Welcome to MESH"

#define TOPIC_NETWORKS \
"### Networks "

#define TOPIC_TRAINING \
"### Training"

struct help
{
        char *help_topic;
        char *help_text;
};

const static struct help hts[] = {

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {"general",             TOPIC_GENERAL},
        {"networks",            TOPIC_NETWORKS},
        {"training",            TOPIC_TRAINING},

        /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
        {NULL,         NULL}
};

void help(char *help_topic);

#endif /* HELP_H */
