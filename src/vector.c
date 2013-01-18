/*
 * vector.c
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

#include "main.h"
#include "math.h"
#include "vector.h"

/*
 * Creates a new vector.
 */

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

/*
 * Disposes a vector.
 */ 

void dispose_vector(struct vector *v)
{
        free(v->elements);
        free(v);
}

/*
 * Copies the values of vector v2 into vector v1.
 */

void copy_vector(struct vector *v1, struct vector *v2)
{
        if(v1->size != v2->size)
                return;

#ifdef _OPENMP
#pragma omp parallel for if(v1->size > OMP_MIN_ITERATIONS)
#endif /* _OPENMP */
        for (int i = 0; i < v1->size; i++)
                v1->elements[i] = v2->elements[i];
}

/*
 * Randomizes the values of a vector using samples
 * from N(mu,sigma).
 */

void randomize_vector(struct vector *v, double mu, double sigma)
{
#ifdef _OPENMP
#pragma omp parallel for if(v->size > OMP_MIN_ITERATIONS)
#endif /* _OPENMP */
        for (int i = 0; i < v->size; i++)
                v->elements[i] = normrand(mu, sigma);
}

/*
 * Sets all vector values to zero.
 */

void zero_out_vector(struct vector *v)
{
#ifdef _OPENMP
#pragma omp parallel for if(v->size > OMP_MIN_ITERATIONS)
#endif /* _OPENMP */
        for (int i = 0; i < v->size; i++)
                v->elements[i] = 0.0;
}

/*
 * Sets all vector cells to a specified value.
 */

void fill_vector_with_value(struct vector *v, double val)
{
#ifdef _OPENMP
#pragma omp parallel for if(v->size > OMP_MIN_ITERATIONS)
#endif /* _OPENMP */
        for (int i = 0; i < v->size; i++)
                v->elements[i] = val;
}

/*
 * Returns the minimum value in a vector.
 */

double vector_minimum(struct vector *v)
{
        double min = v->elements[0];

        for (int i = 0; i < v->size; i++)
                if (v->elements[i] < min)
                        min = v->elements[i];

        return min;
}

/*
 * Returns the maximum values in a vector.
 */

double vector_maximum(struct vector *v)
{
        double max = v->elements[0];

        for (int i = 0; i < v->size; i++)
                if (v->elements[i] > max)
                        max = v->elements[i];

        return max;
}

/*
 * Prints a vector.
 */

void print_vector(struct vector *v)
{
        for (int i = 0; i < v->size; i++)
                printf("%lf\t", v->elements[i]);
        printf("\n");
}
