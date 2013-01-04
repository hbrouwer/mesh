/*
 * error.h
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

#ifndef ERROR_H
#define ERROR_H

#include "network.h"

double error_sum_of_squares(struct group *g, struct vector *t);
void error_sum_of_squares_deriv(struct group *g, struct vector *t);

double error_cross_entropy(struct group *g, struct vector *t);
void error_cross_entropy_deriv(struct group *g, struct vector *t);

double error_divergence(struct group *g, struct vector *t);
void error_divergence_deriv(struct group *g, struct vector *t);

#endif /* ERROR_H */