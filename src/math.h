/*
 * math.h
 *
 * Copyright 2012-2014 Harm Brouwer <me@hbrouwer.eu>
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

#ifndef MATH_H
#define MATH_H

#include "vector.h"

/**************************************************************************
 *************************************************************************/
double minimum(double x, double y);
double maximum(double x, double y);
double sign(double x);

/**************************************************************************
 *************************************************************************/
double normrand(double mu, double sigma);

/**************************************************************************
 *************************************************************************/
double inner_product(struct vector *v1, struct vector *v2);
double harmonic_mean(struct vector *v1, struct vector *v2);
double cosine(struct vector *v1, struct vector *v2);
double tanimoto(struct vector *v1, struct vector *v2);
double dice(struct vector *v1, struct vector *v2);
double pearson_correlation(struct vector *v1, struct vector *v2);

#endif /* MATH_H */
