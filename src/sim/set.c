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

struct element *create_element(struct vector *input, struct vector *target)
{
        struct element *e;

        if (!(e = malloc(sizeof(struct element))))
                goto error_out;
        memset(e, 0, sizeof(struct element));

        e->input = input;
        e->target = target;

        return e;

error_out:
        perror("[create_element()]");
        return NULL;
}

void dispose_element(struct element *e)
{
        if (e->input)
                dispose_vector(e->input);
        if (e->target)
                dispose_vector(e->target);
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
                char *tokens = strtok(buf, " ");

                if (strcmp(tokens, "Input") != 0)
                        continue;

                struct vector *input = create_vector(input_size);
                for (int i = 0; i < input_size; i++) {
                        tokens = strtok(NULL, " ");
                        sscanf(tokens, "%lf", &input->elements[i]);
                }

                struct vector *target = NULL;
                if (tokens = strtok(NULL, " ")) {
                        if (strcmp(tokens, "Target") != 0) {
                                dispose_vector(input);
                                continue;
                        }

                        target = create_vector(output_size);
                        for (int i = 0; i < output_size; i++) {
                                tokens = strtok(NULL, " ");
                                sscanf(tokens, "%lf", &target->elements[i]);
                        }
                }

                s->elements[s->num_elements++] =
                        create_element(input, target);
                if (s->num_elements == s->max_elements)
                        increase_set_size(s);
        }

        fclose(fd);

        return s;

error_out:
        perror("[load_set()]");
        return NULL;
}
