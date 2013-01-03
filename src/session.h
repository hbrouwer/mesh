/*
 * session.h
 *
 * Copyright 2012 Harm Brouwer <me@hbrouwer.eu>
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

#ifndef SESSION_H
#define SESSION_H

#include "network.h"

#define MAX_NETWORKS 10

struct session
{
        struct network_array *networks;
        struct network *anp;
};

struct network_array
{
        int num_elements;           /* number of networks */
        int max_elements;           /* maximum number of networks */
        struct network **elements;  /* the actual networks */
};

struct session *create_session();
void dispose_session(struct session *s);

struct network_array *create_network_array(int max_elements);
void add_to_network_array(struct network_array *ns, struct network *n);
void increase_network_array_size(struct network_array *ns);
void dispose_network_array(struct network_array *ns);

#endif /* SESSION_H */
