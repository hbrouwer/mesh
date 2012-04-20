/*
 * matrix.h
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

#ifndef MATRIX_H
#define MATRIX_H

#include "vector.h"

struct matrix
{
        int rows;                   /* number of rows */
        int cols;                   /* number of columns */
        double **elements;          /* individual values */
};

struct matrix *create_matrix(int rows, int cols);
void dispose_matrix(struct matrix *m);
void copy_matrix(struct matrix *m1, struct matrix *m2);

struct vector *row_to_vector(struct matrix *m, int row);
struct vector *column_to_vector(struct matrix *m, int col);

void randomize_matrix(struct matrix *m, double mu, double sigma);
void binary_randomize_matrix(struct matrix *m);
void zero_out_matrix(struct matrix *m);

void print_matrix(struct matrix *m);

#endif /* MATRIX_H */
