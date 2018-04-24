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
                goto error_file;
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
                                goto error_format;
                        inputs[i] = create_vector(input_size);
                        for (uint32_t j = 0; j < input_size; j++) {
                                /* error: vector too short */
                                if (!(tokens = strtok(NULL, " ")))
                                        goto error_input_vector;
                                /* error: non-numeric unit */                                
                                if (sscanf(tokens, "%lf",
                                        &inputs[i]->elements[j]) != 1)
                                        goto error_input_vector;
                        }
                
                        /* 
                         * Read (optional) target vector, which should be of
                         * the same size as the output vector of the active
                         * network.
                         */
                        if ((tokens = strtok(NULL, " ")) == NULL)
                                continue;
                        if (strcmp(tokens, "Target") != 0)
                                goto error_format;
                        targets[i] = create_vector(output_size);
                        for (uint32_t j = 0; j < output_size; j++) {
                                /* error: vector too short */
                                if (!(tokens = strtok(NULL, " ")))
                                        goto error_target_vector;
                                /* error: non-numeric unit */
                                if (sscanf(tokens, "%lf",
                                        &targets[i]->elements[j]) != 1)
                                        goto error_target_vector;
                                /* error: vector too long */
                                if (j == output_size - 1
                                        && strtok(NULL, " ") != NULL)
                                        goto error_target_vector;
                        }
                }

                /* create an item, and add it to the set */
                item = create_item(name, num_events, meta, inputs, targets);
                add_to_array(s->items, item);
        }
        fclose(fd);

        /* error: emtpy set */
        if (s->items->num_elements == 0) {
                free_set(s);
                goto error_format;
        }

        /* item order equals read order */
        size_t block_size = s->items->num_elements * sizeof(uint32_t);
        if (!(s->order = malloc(block_size)))
                goto error_out;
        memset(s->order, 0, block_size);
        order_set(s);

        return s;

error_file:
        eprintf("Cannot load set - no such file '%s'\n", filename);
        return NULL;
error_format:
        eprintf("Cannot load set - file has incorrect format\n");
        return NULL;        
error_input_vector:
        eprintf("Cannot load set - input vector of incorrect size\n");
        return NULL;
error_target_vector:
        eprintf("Cannot load set - target vector of incorrect size\n");
        return NULL;
error_out:
        perror("[load_set()]");
        return NULL;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Load a set of input and target items. The expected format is:

        [Dimensions I O]

        BeginItem
        Name "name"
        Meta "meta"
        Input # # # Target # #
        Input # # # Target # #
        EndItem

        BeginItem
        Name "name"
        Meta "meta"
        Input # # # Target # #
        EndItem

        BeginItem
        Name "name"
        Meta "meta"
        Input # # #
        Input # # # Target # #
        EndItem

        [...]

where 'name' is an identifier for the item, 'meta' is item-specific meta
information, and '#' are integer or floating point units of the input/target
vectors. Note that target vectors do not need to be present for every input
pattern. The optional "Dimensions I T" specification can be used to override
the dimensions derived from the model (input and output group size).
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

struct set *load_set_nf(char *name, char *filename, uint32_t input_size,
        uint32_t output_size)
{
        struct set *s = create_set(name);

        uint32_t input_dims;
        uint32_t output_dims;

        FILE *fd;
        if (!(fd = fopen(filename, "r")))
                goto error_file;
        char buf[MAX_BUF_SIZE];
        bool fline = true;
        while (fgets(buf, sizeof(buf), fd)) {
                buf[strlen(buf) - 1] = '\0';
                /* comment or blank line */
                switch (buf[0]) {
                case '%':       /* verbose comment */
                        cprintf("\x1b[1m\x1b[36m%s\x1b[0m\n", buf);
                        continue;
                case '#':       /* silent comment */
                case '\0':      /* blank line */
                        continue;
                }

                /* dimensions specification */
                if (fline) {
                        if (sscanf(buf, "Dimensions %d %d",
                                &input_dims, &output_dims) != 2) {
                                if (input_size == 0 || output_size == 0)
                                        goto error_unknown_dimensions;
                                input_dims  = input_size;
                                output_dims = output_size;
                        }
                        fline = false;
                }

                /* item */
                if (strcmp(buf, "BeginItem") == 0) {
                        struct item *item = load_item(fd, input_dims, output_dims);
                        if (item == NULL)
                                goto error_format;
                        add_to_array(s->items, item);
                }
        }
        fclose(fd);

        /* error: emtpy set */
        if (s->items->num_elements == 0) {
                free_set(s);
                goto error_format;
        }

        /* item order equals read order */
        size_t block_size = s->items->num_elements * sizeof(uint32_t);
        if (!(s->order = malloc(block_size)))
                goto error_out;
        memset(s->order, 0, block_size);
        order_set(s);

        return s;

error_file:
        eprintf("Cannot load set - no such file '%s'\n", filename);
        return NULL;
error_unknown_dimensions:
        eprintf("Cannot load set - unknown dimensions\n");
        return NULL;
error_format:
        eprintf("Cannot load set - file has incorrect format\n");
        return NULL;
error_out:
        perror("[load_set()]");
        return NULL;
}

struct item *load_item(FILE *fd, uint32_t input_dims, uint32_t output_dims)
{
        char *name = NULL, *meta = NULL;
        struct array *inputs  = create_array(atype_vectors);
        struct array *targets = create_array(atype_vectors);

        char buf[MAX_BUF_SIZE];
        char arg[MAX_BUF_SIZE];
        while (fgets(buf, sizeof(buf), fd)) {
                buf[strlen(buf) - 1] = '\0';
                /* comment or blank line */
                switch (buf[0]) {
                case '%':       /* verbose comment */
                        cprintf("\x1b[1m\x1b[36m%s\x1b[0m\n", buf);
                        continue;
                case '#':       /* silent comment */
                case '\0':      /* blank line */
                        continue;
                }

                /* item name */
                if (sscanf(buf, "Name \"%[^\"]\"", arg) == 1) {
                        size_t block_size = ((strlen(arg) + 1) * sizeof(char));
                        if (!(name = malloc(block_size)))
                                goto error_out;
                        memset(name, 0, block_size);
                        strncpy(name, arg, strlen(arg));
                }

                /* meta info */
                if (sscanf(buf, "Meta \"%[^\"]\"", arg) == 1) {
                        size_t block_size = ((strlen(arg) + 1) * sizeof(char));
                        if (!(meta = malloc(block_size)))
                                goto error_out;
                        memset(meta, 0, block_size);
                        strncpy(meta, arg, strlen(arg));
                }

                /* end of item */
                if (strcmp(buf, "EndItem") == 0)
                        break;

                /* next line if not an input pattern */
                char *tokens = strtok(buf, " ");
                if (strcmp(tokens, "Input") != 0)
                        continue;
                
                struct vector *input  = create_vector(input_dims);
                struct vector *target = create_vector(output_dims);
                add_to_array(inputs, input);
                add_to_array(targets, target);
                
                for (uint32_t i = 0; i < input_dims; i++) {
                        /* error: vector too short */
                        if (!(tokens = strtok(NULL, " ")))
                                goto error_input_vector;
                        /* error: non-numeric unit */                                
                        if (sscanf(tokens, "%lf", &input->elements[i]) != 1)
                                goto error_input_vector;
                }
                
                /* next line if there is no target pattern */
                if ((tokens = strtok(NULL, " ")) == NULL)
                        continue;
                if (strcmp(tokens, "Target") != 0)
                        continue;

                for (uint32_t i = 0; i < output_dims; i++) {
                        /* error: vector too short */
                        if (!(tokens = strtok(NULL, " ")))
                                goto error_target_vector;
                        /* error: non-numeric unit */
                        if (sscanf(tokens, "%lf", &target->elements[i]) != 1)
                                goto error_target_vector;
                        /* error: vector too long */
                        if (i == output_dims - 1 && strtok(NULL, " ") != NULL)
                                goto error_target_vector;
                }
        }

        /* error: empty set */
        if (inputs->num_elements == 0)
                goto error_format;

        /* input vectors */
        uint32_t num_events = inputs->num_elements;
        struct vector **input_vecs;
        size_t block_size = num_events * sizeof(struct vector *);
        if (!(input_vecs = malloc(block_size)))
                goto error_out;
        memset(input_vecs, 0, block_size);
        for (uint32_t i = 0; i < num_events; i++) {
                input_vecs[i] = create_vector(input_dims);
                copy_vector(input_vecs[i], inputs->elements[i]);
                free_vector(inputs->elements[i]);
        }
        free_array(inputs);

        /* target vectors */
        struct vector **target_vecs;
        if (!(target_vecs = malloc(block_size)))
                goto error_out;
        memset(target_vecs, 0, block_size);
        for (uint32_t i = 0; i < num_events; i++) {
                target_vecs[i] = create_vector(output_dims);
                copy_vector(target_vecs[i], targets->elements[i]);
                free_vector(targets->elements[i]);
        }
        free_array(targets);

        return create_item(name, num_events, meta, input_vecs, target_vecs);

error_input_vector:
        eprintf("Cannot load set - input vector of incorrect size\n");
        return NULL;
error_target_vector:
        eprintf("Cannot load set - target vector of incorrect size\n");
        return NULL;
error_format:
        eprintf("Cannot load set - file has incorrect format\n");
        return NULL;
error_out:
        perror("[load_item()]");
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
