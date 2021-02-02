/*
 * Copyright 2012-2021 Harm Brouwer <me@hbrouwer.eu>
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

#include <stdint.h>

#include "array.h"
#include "network.h"
#include "pprint.h"

struct session
{
        struct array *networks;     /* networks in this session */
        struct network *anp;        /* active network pointer */
        bool pprint;                /* flag for pretty printing */
        enum color_scheme scheme;   /* pretty priniting scheme */
};

struct session *create_session();
void free_session(struct session *s);

void add_network(struct session *s, struct network *n);
void remove_network(struct session *s, struct network *n);
void print_networks(struct session *s);

#endif /* SESSION_H */
