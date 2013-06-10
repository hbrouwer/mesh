/*
 * set.h
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

#ifndef SET_H
#define SET_H

#include "array.h"
#include "vector.h"

struct set
{
        char *name;               /* name of this set */
        struct array *items;      /* items */
        int *order;               /* order in which to present elements */
};

struct item
{
        char *name;               /* name of this element */
        int num_events;           /* number of events */
        char *meta;               /* meta information */
        struct vector **inputs;   /* input vectors */
        struct vector **targets;  /* target vectors */
};

/**************************************************************************
 *************************************************************************/
struct set *create_set(char *name);
void dispose_set(struct set *s);

/**************************************************************************
 *************************************************************************/
struct item *create_item(char *name, int num_events, char *meta,
                struct vector **inputs, struct vector **targets);
void dispose_item(struct item *item);

/**************************************************************************
 *************************************************************************/
struct set *load_set(char *name, char *filename, int input_size, int output_size);

/**************************************************************************
 *************************************************************************/
void order_set(struct set *s);
void permute_set(struct set *s);
void randomize_set(struct set *s);

#endif /* SET_H */
