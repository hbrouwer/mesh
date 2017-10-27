/*
 * Copyright 2012-2017 Harm Brouwer <me@hbrouwer.eu>
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

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Schraudolph's approximation of the exponentional function. See:

Schraudolph, N. N. (1999). A fast, compact approximation of the
        exponentional function. Neural Computation, 11, 854-862.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#ifdef FAST_EXP
__attribute__((unused))
static union
{
        double d;
        struct
        {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
                int j, i;
#else
                int i, j;
#endif
        } n;
} _eco;

#define EXP_A (1048576 / M_LN2)
#define EXP_C 60801

#define EXP_APPROX(x) (_eco.n.i = EXP_A * (x) + (1072693248 - EXP_C),_eco.d)

double fast_exp(double x);
#endif /* FAST_EXP */

double minimum(double x, double y);
double maximum(double x, double y);
double sign(double x);

double normrand(double mu, double sigma);

double runge_kutta4(double (*f)(double, double), double h, double xn,
       double yn);

double euclidean_norm(struct vector *v);

double inner_product(struct vector *v1, struct vector *v2);
double harmonic_mean(struct vector *v1, struct vector *v2);
double cosine(struct vector *v1, struct vector *v2);
double tanimoto(struct vector *v1, struct vector *v2);
double dice(struct vector *v1, struct vector *v2);
double pearson_correlation(struct vector *v1, struct vector *v2);

#endif /* MATH_H */
