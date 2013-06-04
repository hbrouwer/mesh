/*
 * array.c
 *
 * Copyright 2012, 2013 Harm Brouwer <me@hbrouwer.eu>
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

#include "array.h"
#include "main.h"
#include "network.h"
#include "session.h"
#include "set.h"

struct array *create_array(int type)
{
        struct array *a;
        if (!(a = malloc(sizeof(struct array))))
                goto error_out;
        memset(a, 0, sizeof(struct array));

        a->type = type;
        a->num_elements = 0;
        a->max_elements = MAX_ARRAY_ELEMENTS;

        int block_size = a->max_elements * sizeof(void *);
        if (!(a->elements = malloc(block_size)))
                goto error_out;

        return a;

error_out:
        perror("[create_array()]");
        return NULL;
}

void add_to_array(struct array *a, void *e)
{
        a->elements[a->num_elements++] = e;
        if (a->num_elements == a->max_elements)
                increase_array_size(a);
}

void remove_from_array(struct array *a, void *e)
{
        int i;
        for (i = 0; i < a->num_elements; i++)
                if (a->elements[i] == e)
                        break;
        for (int j = i; j < a->num_elements - 1; j++)
                a->elements[j] = a->elements[j + 1];
        a->elements[a->num_elements - 1] = NULL;
        a->num_elements--;
}

void increase_array_size(struct array *a)
{
        a->max_elements = a->max_elements + MAX_ARRAY_ELEMENTS;

        int block_size = a->max_elements * sizeof(void *);
        if (!(a->elements = realloc(a->elements, block_size)))
                goto error_out;

        for (int i = a->num_elements; i < a->max_elements; i++)
                a->elements[i] = NULL;

        return;

error_out:
        perror("[increase_array_size()]");
        return;
}

void dispose_array(struct array *a)
{
        free(a->elements);
        free(a);
}

void *find_array_element_by_name(struct array *a, char *name)
{
        for (int i = 0; i < a->num_elements; i++) {
                void *e = a->elements[i];

                if (a->type == TYPE_NETWORKS) {
                        struct network *n = (struct network *)e;
                        if (strcmp(n->name, name) == 0)
                                return e;
                }
                if (a->type == TYPE_GROUPS) {
                        struct group *g = (struct group *)e;
                        if (strcmp(g->name, name) == 0)
                                return e;
                }
                if (a->type == TYPE_SETS) {
                        struct set *s = (struct set *)e;
                        if (strcmp(s->name, name) == 0)
                                return e;
                }
                if (a->type == TYPE_ITEMS) {
                        struct item *item = (struct item *)e;
                        if (strcmp(item->name, name) == 0)
                                return e;
                }
        }

        return NULL;
}
