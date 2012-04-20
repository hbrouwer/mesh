/*
 * vector.c
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
#include "math.h"
#include "vector.h"

struct vector *create_vector(int size)
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

void dispose_vector(struct vector *v)
{
        free(v->elements);
        free(v);
}

void copy_vector(struct vector *v1, struct vector *v2)
{
        if(v1->size != v2->size)
                return;

        for (int i = 0; i < v1->size; i++)
                v1->elements[i] = v2->elements[i];
}

void randomize_vector(struct vector *v, double mu, double sigma)
{
        for (int i = 0; i < v->size; i++)
                v->elements[i] = normrand(mu, sigma);
}

void zero_out_vector(struct vector *v)
{
        for (int i = 0; i < v->size; i++)
                v->elements[i] = 0.0;
}

void print_vector(struct vector *v)
{
        for (int i = 0; i < v->size; i++)
                printf("%.2lf\t", v->elements[i]);
        printf("\n");
}
