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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "defaults.h"
#include "main.h"
#include "network.h"
#include "session.h"

struct session *create_session()
{
        struct session *s;
        if (!(s = malloc(sizeof(struct session))))
                goto error_out;
        memset(s, 0, sizeof(struct session));

        s->networks = create_array(atype_networks);
        s->pprint   = DEFAULT_PRETTY_PRINTING;
        s->scheme   = DEFAULT_COLOR_SCHEME;

        return s;

error_out:
        perror("[create_session()]");
        return NULL;
}

void free_session(struct session *s)
{
        for (uint32_t i = 0; i < s->networks->num_elements; i++)
                free_network(s->networks->elements[i]);
        free_array(s->networks);
        free(s);
}

void add_network(struct session *s, struct network *n)
{
        add_to_array(s->networks, n);
        s->anp = n;
}

void remove_network(struct session *s, struct network *n)
{
        /*
         * If the network to be removed is the active network, try finding
         * another active network.
         */
        if (n == s->anp) {
                s->anp = NULL;
                for (uint32_t i = 0; i < s->networks->num_elements; i++) {
                        struct network *anp = s->networks->elements[i];
                        if (anp != NULL && anp != n)
                                s->anp = anp;
                }
        }
        /* remove network */
        remove_from_array(s->networks, n);
        free_network(n);
}

void print_networks(struct session *s)
{
        for (uint32_t i = 0; i < s->networks->num_elements; i++) {
                struct network *n = s->networks->elements[i];
                cprintf("* %d: %s", i + 1, n->name);
                n == s->anp ? cprintf(" :: active network\n")
                            : cprintf("\n");
        }       
}
