/*
 * Copyright 2012-2017 Harm Brouwer <me@hbrouwer.eu>
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

#ifndef ARRAY_H
#define ARRAY_H

#include <stdint.h>

#define MAX_ARRAY_ELEMENTS 10

#define ATYPE_NETWORKS   0
#define ATYPE_GROUPS     1
#define ATYPE_PROJS      2
#define ATYPE_SETS       3
#define ATYPE_ITEMS      4

struct array
{
        uint32_t type;              /* array type */
        uint32_t num_elements;      /* number of elements in the array */
        uint32_t max_elements;      /* max number of elements */
        void **elements;            /* elements */
};

struct array *create_array(uint32_t type);
void add_to_array(struct array *a, void *e);
void remove_from_array(struct array *a, void *e);
void increase_array_size(struct array *a);
void dispose_array(struct array *a);
void *find_array_element_by_name(struct array *a, char *name);

#endif /* ARRAY_H */
