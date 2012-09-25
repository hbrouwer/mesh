/*
 * set.c
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

#include "main.h"
#include "set.h"

struct set *create_set(int max_elements)
{
        struct set *s;
        if (!(s = malloc(sizeof(struct set))))
                goto error_out;
        memset(s, 0, sizeof(struct set));

        s->num_elements = 0;
        s->max_elements = max_elements;

        int block_size = s->max_elements * sizeof(struct element *);
        if (!(s->elements = malloc(block_size)))
                goto error_out;
        memset(s->elements, 0, block_size);

        return s;

error_out:
        perror("[create_set()])");
        return NULL;
}

void increase_set_size(struct set *s)
{
        s->max_elements = s->max_elements + MAX_ELEMENTS;

        int block_size = s->max_elements * sizeof(struct element *);
        if (!(s->elements = realloc(s->elements, block_size)))
                goto error_out;
        for(int i = s->num_elements; i < s->max_elements; i++)
                s->elements[i] = NULL;

        return;

error_out:
        perror("[increase_set_size()]");
        return;
}

void dispose_set(struct set *s)
{
        for (int i = 0; i < s->max_elements; i++)
                if (s->elements[i])
                        dispose_element(s->elements[i]);
        free(s->elements);
        free(s);
}

struct element *create_element(char *name, int num_events, 
                struct vector **inputs, struct vector **targets)
{
        struct element *e;

        if (!(e = malloc(sizeof(struct element))))
                goto error_out;
        memset(e, 0, sizeof(struct element));

        e->name = name;
        e->num_events = num_events;
        e->inputs = inputs;
        e->targets = targets;

        return e;

error_out:
        perror("[create_element()]");
        return NULL;
}

void dispose_element(struct element *e)
{
        for (int i = 0; i < e->num_events; i++) {
                if (e->inputs[i])
                        dispose_vector(e->inputs[i]);
                if (e->targets[i])
                        dispose_vector(e->targets[i]);
        }

        free(e->inputs);
        free(e->targets);

        free(e);
}

struct set *load_set(char *filename, int input_size, int output_size)
{
        struct set *s = create_set(MAX_ELEMENTS);

        FILE *fd;
        if (!(fd = fopen(filename, "r")))
                goto error_out;

        char buf[4096];
        while (fgets(buf, sizeof(buf), fd)) {
                char tmp[64];
                int num_events;

                if (!(sscanf(buf, "Name \"%[^\"]\" %d", tmp, &num_events)))
                        continue;

                char *name;
                int block_size = ((strlen(tmp) + 1) * sizeof(char));
                if (!(name = malloc(block_size)))
                        goto error_out;
                memset(name, 0, block_size);
                strncpy(name, tmp, strlen(tmp));

                struct vector **inputs;
                block_size = num_events * sizeof(struct vector *);
                if (!(inputs = malloc(block_size)))
                        goto error_out;
                memset(inputs, 0, block_size);
                struct vector **targets;
                if (!(targets = malloc(block_size)))
                        goto error_out;
                memset(targets, 0, block_size);

                for (int i = 0; i < num_events; i++) {
                        if (!(fgets(buf, sizeof(buf), fd)))
                                goto error_out;
                        char *tokens = strtok(buf, " ");

                        if (strcmp(tokens, "Input") == 0) {
                                inputs[i] = create_vector(input_size);
                                for (int j = 0; j < input_size; j++) {
                                        if (!(tokens = strtok(NULL, " ")))
                                                goto error_out;
                                        if (!sscanf(tokens, "%lf", &inputs[i]->elements[j]))
                                                goto error_out;
                                }
                        }

                        if ((tokens = strtok(NULL, " ")) != NULL) {
                                targets[i] = create_vector(output_size);
                                if (strcmp(tokens, "Target") == 0) {
                                        for (int j = 0; j < output_size; j++) {
                                                if (!(tokens = strtok(NULL, " ")))
                                                        goto error_out;
                                                if (!sscanf(tokens, "%lf", &targets[i]->elements[j]))
                                                        goto error_out;
                                        }
                                }
                        }
                }

                s->elements[s->num_elements++] =
                        create_element(name, num_events, inputs, targets);
                if (s->num_elements == s->max_elements)
                        increase_set_size(s);
        }

        fclose(fd);

        return s;

error_out:
        perror("[load_set()]");
        return NULL;
}

struct set *permute_set(struct set *s)
{
        struct set *ps = create_set(s->num_elements);
        ps->num_elements = s->num_elements;

        for (int i = 0; i < s->num_elements; i++) {
                int pe = ((double)rand() / (double)RAND_MAX)
                        * (s->num_elements);
                struct element *e = s->elements[pe];

                bool duplicate = false;
                for (int j = 0; j < i; j++)
                        if (ps->elements[j] == e)
                                duplicate = true;

                if (duplicate)
                        i--;
                else
                        ps->elements[i] = e;
        }

        return ps;
}

struct set *randomize_set(struct set *s)
{
        struct set *rs = create_set(s->num_elements);
        rs->num_elements = s->num_elements;

        for (int i = 0; i < s->num_elements; i++) {
                int re = ((double)rand() / (double)RAND_MAX)
                        * (s->num_elements);
                rs->elements[i] = s->elements[re];
        }

        return rs;
}
