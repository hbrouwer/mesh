/*
 * set.h
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

#ifndef SET_H
#define SET_H

#include "vector.h"

#define MAX_ELEMENTS 100

struct set
{
        int num_elements;           /* number of set elements */
        int max_elements;           /* maximum number of set elements */
        struct element **elements;  /* set elements */
};

struct element
{
        struct vector *input;       /* input vector */
        struct vector *target;      /* target vector */
};

struct set *create_set(int max_elements);
void increase_set_size(struct set *s);
void dispose_set(struct set *s);

struct element *create_element(struct vector *input, struct vector *target);
void dispose_element(struct element *e);

struct set *load_set(char *filename, int input_size, int output_size);

#endif /* SET_H */
