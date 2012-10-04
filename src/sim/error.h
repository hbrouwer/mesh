/*
 * error.h
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

#ifndef ERROR_H
#define ERROR_H

#include "network.h"

double error_sum_of_squares(struct vector *o, struct vector *t);
struct vector *error_sum_of_squares_deriv(struct vector *o, struct vector *t);

double error_cross_entropy(struct vector *o, struct vector *t);
struct vector *error_cross_entropy_deriv(struct vector *o, struct vector *t);

double error_divergence(struct vector *o, struct vector *t);
struct vector *error_divergence_deriv(struct vector *o, struct vector *t);

#endif /* ERROR_H */
