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

#ifndef ERROR_H
#define ERROR_H

#include "network.h"
#include "vector.h"

#define LARGE_VALUE 1e8
#define SMALL_VALUE 1e-8

double adjust_target(double y, double d, double tr, double zr);

/* sum of squares */
double err_fun_sum_of_squares(struct network *n, struct group *g,
        struct vector *t);
void err_fun_sum_of_squares_deriv(struct network *n, struct group *g,
        struct vector *t);

/* cross entropy */
double err_fun_cross_entropy(struct network *n, struct group *g,
        struct vector *t);
void err_fun_cross_entropy_deriv(struct network *n, struct group *g,
        struct vector *t);

/* divergence */
double err_fun_divergence(struct network *n, struct group *g,
        struct vector *t);
void err_fun_divergence_deriv(struct network *n, struct group *g,
        struct vector *t);

#endif /* ERROR_H */
