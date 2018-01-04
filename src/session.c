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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "defaults.h"
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
