/*
 * matrix.c
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

#include <math.h>

#include "main.h"
#include "math.h"
#include "matrix.h"

/*
 * Creates a new matrix.
 */

struct matrix *create_matrix(int rows, int cols)
{
        struct matrix *m;
        if (!(m = malloc(sizeof(struct matrix))))
                goto error_out;
        memset(m, 0, sizeof(struct matrix));

        m->rows = rows;
        m->cols = cols;

        if (!(m->elements = malloc(m->rows * sizeof(double *))))
                goto error_out;
        memset(m->elements, 0, m->rows * sizeof(double *));

        for (int i = 0; i < m->rows; i++) {
                if (!(m->elements[i] = malloc(m->cols * sizeof(double))))
                        goto error_out;
                memset(m->elements[i], 0, m->cols * sizeof(double));
        }

        return m;

error_out:
        perror("[create_matrix()]");
        return NULL;
}

/*
 * Disposes a matrix.
 */

void dispose_matrix(struct matrix *m)
{
        for (int i = 0; i < m->rows; i++)
                free(m->elements[i]);
        free(m->elements);
        free(m);
}

/*
 * Copies the values of matrix m1 into matrix m2.
 */

void copy_matrix(struct matrix *m1, struct matrix *m2)
{
        if(m1->rows != m2->rows || m1->cols != m2->cols)
                return;

#ifdef _OPENMP
#pragma omp parallel for if(m1->rows > OMP_MIN_ITERATIONS)
#endif /* _OPENMP */
        for (int i = 0; i < m1->rows; i++)
                for (int j = 0; j < m1->cols; j++)
                        m1->elements[i][j] = m2->elements[i][j];
}

/*
 * Creates a vector out of the values in a matrix row.
 */

struct vector *row_to_vector(struct matrix *m, int row)
{
        struct vector *v = create_vector(m->cols);

#ifdef _OPENMP
#pragma omp parallel for if(m->cols > OMP_MIN_ITERATIONS)
#endif /* _OPENMP */
        for (int i = 0; i < m->cols; i++)
                v->elements[i] = m->elements[row][i];

        return v;
}

/*
 * Creates a vector out of the values in a matrix column.
 */

struct vector *column_to_vector(struct matrix *m, int col)
{
        struct vector *v = create_vector(m->rows);

#ifdef _OPENMP
#pragma omp parallel for if(m->rows > OMP_MIN_ITERATIONS)
#endif /* _OPENMP */
        for (int i = 0; i < m->rows; i++)
                v->elements[i] = m->elements[i][col];

        return v;
}

/*
 * Randomizes the values of a matrix using samples
 * from N(mu,sigma).
 */

void randomize_matrix(struct matrix *m, double mu, double sigma)
{
//#ifdef _OPENMP
//#pragma omp parallel for if(m->rows > OMP_MIN_ITERATIONS)
//#endif /* _OPENMP */
        for (int i = 0; i < m->rows; i++)
                for (int j = 0; j < m->cols; j++)
                        m->elements[i][j] = normrand(mu, sigma);
}

/*
 * Randomly fills a matrix with binary values.
 */

void binary_randomize_matrix(struct matrix *m)
{
//#ifdef _OPENMP
//#pragma omp parallel for if(m->rows > OMP_MIN_ITERATIONS)
//#endif /* _OPENMP */
        for (int i = 0; i < m->rows; i++)
                for (int j = 0; j < m->cols; j++)
                        m->elements[i][j] = round((float)rand() / RAND_MAX);
}

/*
 * Sets all matrix values to zero.
 */

void zero_out_matrix(struct matrix *m)
{
#ifdef _OPENMP
#pragma omp parallel for if(m->rows > OMP_MIN_ITERATIONS)
#endif /* _OPENMP */
        for (int i = 0; i < m->rows; i++)
                for (int j = 0; j < m->cols; j++)
                        m->elements[i][j] = 0.0;
}

/*
 * Sets all matrix cells to a specified value.
 */ 

void fill_matrix_with_value(struct matrix *m, double val)
{
#ifdef _OPENMP
#pragma omp parallel for if(m->rows > OMP_MIN_ITERATIONS)
#endif /* _OPENMP */
        for (int i = 0; i < m->rows; i++)
                for (int j = 0; j < m->cols; j++)
                        m->elements[i][j] = val;
}

/*
 * Returns the minimum value in a matrix.
 */

double matrix_minimum(struct matrix *m)
{
        double min = m->elements[0][0];

        for (int i = 0; i < m->rows; i++)
                for (int j = 0; j < m->cols; j++)
                        if (m->elements[i][j] < min)
                                min = m->elements[i][j];

        return min;
}

/*
 * Returns the maximum value in a matrix.
 */

double matrix_maximum(struct matrix *m)
{
        double max = m->elements[0][0];

        for (int i = 0; i < m->rows; i++)
                for (int j = 0; j < m->cols; j++)
                        if (m->elements[i][j] > max)
                                max = m->elements[i][j];

        return max;
}

/*
 * Prints a matrix.
 */

void print_matrix(struct matrix *m)
{
        for (int i = 0; i < m->rows; i++) {
                for (int j = 0; j < m->cols; j++)
                        printf("%lf\t", m->elements[i][j]);
                printf("\n");
        }
}
