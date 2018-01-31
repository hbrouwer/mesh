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

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "main.h"
#include "set.h"

struct set *create_set(char *name)
{
        struct set *s;
        if (!(s = malloc(sizeof(struct set))))
                goto error_out;
        memset(s, 0, sizeof(struct set));
        size_t block_size = (strlen(name) + 1) * sizeof(char);
        if (!(s->name = malloc(block_size)))
                goto error_out;
        memset(s->name, 0, block_size);
        strncpy(s->name, name, strlen(name));

        s->items = create_array(atype_items);

        return s;

error_out:
        perror("[create_set()])");
        return NULL;
}

void free_set(struct set *s)
{
        free(s->name);
        for (uint32_t i = 0; i < s->items->num_elements; i++)
                free_item(s->items->elements[i]);
        free_array(s->items);
        free(s->order);
        free(s);
}

struct item *create_item(char *name, uint32_t num_events, char *meta,
        struct vector **inputs, struct vector **targets)
{
        struct item *item;
        if (!(item = malloc(sizeof(struct item))))
                goto error_out;
        memset(item, 0, sizeof(struct item));

        item->name       = name;
        item->num_events = num_events;
        item->meta       = meta;
        item->inputs     = inputs;
        item->targets    = targets;

        return item;

error_out:
        perror("[create_element()]");
        return NULL;
}

void free_item(struct item *item)
{
        free(item->name);
        free(item->meta);
        for (uint32_t i = 0; i < item->num_events; i++) {
                if (item->inputs[i])  free_vector(item->inputs[i]);
                if (item->targets[i]) free_vector(item->targets[i]);
        }
        free(item->inputs);
        free(item->targets);
        free(item);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Load a set of input and target items. Currently, the expected format is:

        Item "name" num_events "meta"
        Input # # # Target # #
        Input # # # Target # #

        Item "name" num_events "meta"
        Input # # # Target # #

        Item "name" num_events "meta"
        Input # # #
        Input # # # Target # #

        [...]

where 'name' is an identifier for the item, 'num_events' is the number of
input (and target) events, 'meta' is item-specific meta information, and '#'
are integer or floating point units of the input/target vectors. Note that
target vectors do not need to be present for every input pattern.

TODO: Adopt a less spartan file format.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

struct set *load_set(char *name, char *filename, uint32_t input_size,
        uint32_t output_size)
{
        struct set *s = create_set(name);

        FILE *fd;
        if (!(fd = fopen(filename, "r")))
                goto error_out;
        char buf[MAX_BUF_SIZE];
        while (fgets(buf, sizeof(buf), fd)) { 
                char arg1[MAX_BUF_SIZE];        /* item name */
                uint32_t num_events;            /* number of events */
                char arg2[MAX_BUF_SIZE];        /* meta information */
                struct item *item;

                /* skip line, if it does not introduce an item */
                if (sscanf(buf, "Item \"%[^\"]\" %d \"%[^\"]\"",
                        arg1, &num_events, arg2) != 3)
                        continue;

                /* item name */
                char *name;
                size_t block_size = ((strlen(arg1) + 1) * sizeof(char));
                if (!(name = malloc(block_size)))
                        goto error_out;
                memset(name, 0, block_size);
                strncpy(name, arg1, strlen(arg1));

                /* meta information */
                char *meta;
                block_size = ((strlen(arg2) + 1) * sizeof(char));
                if (!(meta = malloc(block_size)))
                        goto error_out;
                memset(meta, 0, block_size);
                strncpy(meta, arg2, strlen(arg2));

                /* input vectors */
                struct vector **inputs;
                block_size = num_events * sizeof(struct vector *);
                if (!(inputs = malloc(block_size)))
                        goto error_out;
                memset(inputs, 0, block_size);

                /* target vectors */
                struct vector **targets;
                if (!(targets = malloc(block_size)))
                        goto error_out;
                memset(targets, 0, block_size);

                /* read input and target vectors */
                for (uint32_t i = 0; i < num_events; i++) {
                        if (!(fgets(buf, sizeof(buf), fd)))
                                goto error_out;
                        char *tokens = strtok(buf, " ");

                        /* 
                         * Read input vector, which should be of the same
                         * size as the input vector of the active network.
                         */
                        if (strcmp(tokens, "Input") != 0)
                                goto error_out;
                        inputs[i] = create_vector(input_size);
                        for (uint32_t j = 0; j < input_size; j++) {
                                if (!(tokens = strtok(NULL, " ")))
                                        goto error_out;
                                if (sscanf(tokens, "%lf",
                                        &inputs[i]->elements[j]) != 1)
                                        goto error_out;
                        }
                
                        /* 
                         * Read (optional) target vector, which should be of
                         * the same size as the output vector of the active
                         * network.
                         */
                        if ((tokens = strtok(NULL, " ")) == NULL)
                                continue;
                        targets[i] = create_vector(output_size);
                        for (uint32_t j = 0; j < output_size; j++) {
                                if (!(tokens = strtok(NULL, " ")))
                                        goto error_out;
                                if (sscanf(tokens, "%lf",
                                        &targets[i]->elements[j]) != 1)
                                        goto error_out;
                        }
                }

                /* create an item, and add it to the set */
                item = create_item(name, num_events, meta, inputs, targets);
                add_to_array(s->items, item);
        }
        fclose(fd);

        /* item order equals read order */
        size_t block_size = s->items->num_elements * sizeof(uint32_t);
        if (!(s->order = malloc(block_size)))
                goto error_out;
        memset(s->order, 0, block_size);
        order_set(s);


        return s;

error_out:
        perror("[load_set()]");
        return NULL;
}

void order_set(struct set *s)
{
        for (uint32_t i = 0; i < s->items->num_elements; i++)
                s->order[i] = i;
}

void permute_set(struct set *s)
{
        for (uint32_t i = 0; i < s->items->num_elements; i++) {
                uint32_t pe = ((double)rand() / (double)RAND_MAX)
                        * (s->items->num_elements);
                bool duplicate = false;
                for (uint32_t j = 0; j < i; j++)
                        if (s->order[j] == pe)
                                duplicate = true;
                if (duplicate)
                        i--;
                else
                        s->order[i] = pe;
        }
}

void randomize_set(struct set *s)
{
        for (uint32_t i = 0; i < s->items->num_elements; i++) {
                uint32_t re = ((double)rand() / (double)RAND_MAX)
                        * (s->items->num_elements);
                s->order[i] = re;
        }
}
