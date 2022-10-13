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

#include "array.h"
#include "network.h"
#include "set.h"

struct array *create_array(enum array_type type)
{
        struct array *a;
        if (!(a = malloc(sizeof(struct array))))
                goto error_out;
        memset(a, 0, sizeof(struct array));

        a->type         = type;
        a->num_elements = 0;
        a->max_elements = MAX_ARRAY_ELEMENTS;

        size_t block_size = a->max_elements * sizeof(void *);
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
        uint32_t i;
        for (i = 0; i < a->num_elements; i++)
                if (a->elements[i] == e)
                        break;
        for (uint32_t j = i; j < a->num_elements - 1; j++)
                a->elements[j] = a->elements[j + 1];
        a->elements[a->num_elements - 1] = NULL;
        a->num_elements--;

        if (a->num_elements < a->max_elements - MAX_ARRAY_ELEMENTS)
                decrease_array_size(a);
}

void increase_array_size(struct array *a)
{
        a->max_elements = a->max_elements + MAX_ARRAY_ELEMENTS;

        /* increase array size */
        size_t block_size = a->max_elements * sizeof(void *);
        if (!(a->elements = realloc(a->elements, block_size)))
                goto error_out;

        /* zero out all additional cells */
        for (uint32_t i = a->num_elements; i < a->max_elements; i++)
                a->elements[i] = NULL; 
        
        return;

error_out:
        perror("[increase_array_size()]");
        return;
}

void decrease_array_size(struct array *a)
{
        a->max_elements = a->max_elements - MAX_ARRAY_ELEMENTS;
        
        /* decrease array size */
        size_t block_size = a->max_elements * sizeof(void *);
        if (!(a->elements = realloc(a->elements, block_size)))
                goto error_out;

        return;

error_out:
        perror("[decrease_array_size()]");
        return;       
}

void free_array(struct array *a)
{
        free(a->elements);
        free(a);
}

/*
 * Note: Projections are not addressable by name.
 * 
 * Note: Items can have names that are not set.
 */
void *find_array_element_by_name(struct array *a, char *name)
{
        if (a == NULL)
                return NULL;

        for (uint32_t i = 0; i < a->num_elements; i++) {
                void *e = a->elements[i];
                if (a->type == atype_networks) {
                        struct network *n = (struct network *)e;
                        if (strcmp(n->name, name) == 0)
                                return e;
                }
                else if (a->type == atype_groups) {
                        struct group *g = (struct group *)e;
                        if (strcmp(g->name, name) == 0)
                                return e;
                }
                else if (a->type == atype_sets) {
                        struct set *s = (struct set *)e;
                        if (strcmp(s->name, name) == 0)
                                return e;
                }
                else if (a->type == atype_items) {
                        struct item *item = (struct item *)e;
                        if (item->name && strcmp(item->name, name) == 0)
                                return e;
                }
        }

        return NULL;
}
