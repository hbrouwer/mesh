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

#ifndef VECTOR_H
#define VECTOR_H

#include <stdint.h>

struct vector
{
        uint32_t size;              /* vector size */
        double *elements;           /* elements */
};

struct vector *create_vector(uint32_t size);
void free_vector(struct vector *v);
void copy_vector(struct vector *v1, struct vector *v2);

void zero_out_vector(struct vector *v);
void fill_vector_with_value(struct vector *v, double val);

double vector_minimum(struct vector *v);
double vector_maximum(struct vector *v);

void print_vector(struct vector *v);

#endif /* VECTOR_H */
