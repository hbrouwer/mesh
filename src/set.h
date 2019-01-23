/*
 * Copyright 2012-2019 Harm Brouwer <me@hbrouwer.eu>
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

#include <stdio.h>
#include <stdint.h>

#include "array.h"
#include "pprint.h"
#include "vector.h"

/* set */
struct set
{
        char *name;                 /* name of this set */
        struct array *items;        /* items */
        uint32_t *order;            /* order in which to present items */
};

/* item */
struct item
{
        char *name;                 /* name of this item */
        uint32_t num_events;        /* number of events */
        char *meta;                 /* meta information */
        struct vector **inputs;     /* input vectors */
        struct vector **targets;    /* target vectors */
};

struct set *create_set(char *name);
void free_set(struct set *s);

struct item *create_item(char *name, char *meta, uint32_t num_events,
        struct vector **inputs, struct vector **targets);
void free_item(struct item *item);
void print_items(struct set *set);
void print_item(struct item *item, bool pprint, enum color_scheme scheme);

struct set *load_legacy_set(char *name, char *filename, uint32_t input_size,
        uint32_t output_size);
struct set *load_set(char *name, char *filename, uint32_t input_size,
        uint32_t output_size);
struct item *load_item(FILE *fd, uint32_t input_dims, uint32_t output_dims);

void order_set(struct set *s);
void permute_set(struct set *s);
void randomize_set(struct set *s);

#endif /* SET_H */
