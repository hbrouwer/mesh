/*
 * Copyright 2012-2022 Harm Brouwer <me@hbrouwer.eu>
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

#include <stdint.h>

struct matrix
{
        uint32_t rows;                  /* number of rows */
        uint32_t cols;                  /* number of columns */
        double **elements;              /* elements */
};

struct matrix *create_matrix(uint32_t rows, uint32_t cols);
void free_matrix(struct matrix *m);
void copy_matrix(struct matrix *sm, struct matrix *dm);

void zero_out_matrix(struct matrix *m);
void fill_matrix_with_value(struct matrix *m, double val);

double matrix_minimum(struct matrix *m);
double matrix_maximum(struct matrix *m);

void print_matrix(struct matrix *m);

#endif /* MATRIX_H */
