/*
 * Copyright 2012-2020 Harm Brouwer <me@hbrouwer.eu>
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

#include "main.h"
#include "vector.h"

struct vector *create_vector(uint32_t size)
{
        struct vector *v;
        if (!(v = malloc(sizeof(struct vector))))
                goto error_out;
        memset(v, 0, sizeof(struct vector));

        v->size = size;
        if (!(v->elements = malloc(v->size * sizeof(double))))
                goto error_out;
        memset(v->elements, 0, v->size * sizeof(double));

        return v;
        
error_out:
        perror("[create_vector()]");
        return NULL;
}

void free_vector(struct vector *v)
{
        free(v->elements);
        free(v);
}

void copy_vector(struct vector *sv, struct vector *dv)
{
        if(sv->size != dv->size)
                return;

        for (uint32_t i = 0; i < sv->size; i++)
                dv->elements[i] = sv->elements[i];
}

void zero_out_vector(struct vector *v)
{
        for (uint32_t i = 0; i < v->size; i++)
                v->elements[i] = 0.0;
}

void fill_vector_with_value(struct vector *v, double val)
{
        for (uint32_t i = 0; i < v->size; i++)
                v->elements[i] = val;
}

double vector_minimum(struct vector *v)
{
        double min = v->elements[0];

        for (uint32_t i = 0; i < v->size; i++)
                if (v->elements[i] < min)
                        min = v->elements[i];

        return min;
}

double vector_maximum(struct vector *v)
{
        double max = v->elements[0];

        for (uint32_t i = 0; i < v->size; i++)
                if (v->elements[i] > max)
                        max = v->elements[i];

        return max;
}

void print_vector(struct vector *v)
{
        cprintf("[ ");
        for (uint32_t i = 0; i < v->size; i++) {
                if (i > 0 && i < v->size)
                        cprintf(", ");
                cprintf("%lf", v->elements[i]);
        }
        cprintf(" ]\n");
}
