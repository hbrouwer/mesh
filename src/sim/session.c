/*
 * session.c
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

#include "session.h"

struct session *create_session()
{
        struct session *s;
        if (!(s = malloc(sizeof(struct session))))
                goto error_out;
        memset(s, 0, sizeof(struct session));

        s->networks = create_network_array(MAX_NETWORKS);

        return s;

error_out:
        perror("[create_session()]");
        return NULL;
}

void dispose_session(struct session *s)
{
        dispose_network_array(s->networks);
        free(s);
}

struct network_array *create_network_array(int max_elements)
{
        struct network_array *ns;
        if (!(ns = malloc(sizeof(struct network_array))))
                goto error_out;
        memset(ns, 0, sizeof(struct network_array));

        ns->num_elements = 0;
        ns->max_elements = max_elements;

        int block_size = ns->max_elements * sizeof(struct network *);
        if (!(ns->elements = malloc(block_size)))
                goto error_out;
        memset(ns->elements, 0, block_size);

        return ns;

error_out:
        perror("[create_network_array()]");
        return NULL;
}

void add_to_network_array(struct network_array *ns, struct network *n)
{
        ns->elements[ns->num_elements++] = n;
        if (ns->num_elements == ns->max_elements)
                increase_network_array_size(ns);
}

void increase_network_array_size(struct network_array *ns)
{
        ns->max_elements = ns->max_elements + MAX_NETWORKS;

        int block_size = ns->max_elements * sizeof(struct network *);
        if (!(ns->elements = realloc(ns->elements, block_size)))
                goto error_out;
        for (int i = ns->num_elements; i < ns->max_elements; i++)
                ns->elements[i] = NULL;

        return;

error_out:
        perror("[increase_network_array_size()]");
        return;
}

void dispose_network_array(struct network_array *ns)
{
        for (int i = 0; i < ns->max_elements; i++)
                if (ns->elements[i])
                        dispose_network(ns->elements[i]);
        free(ns->elements);
        free(ns);
}
