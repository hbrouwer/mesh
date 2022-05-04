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

#include <math.h>

#include "error.h"
#include "main.h"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Adjust a unit's target based on the target radius and zero-error radius. If
a unit's activation is within the target or zero-error radius of the target,
set its target to equal this activation, such that the error for this unit
will be zero. Otherwise, adjust the target in the direction of the unit's
activation by the target radius.

Formulas adapted from LENS (Rohde, 1999) source code.

Rohde, D. L. T. (1999). LENS: the light, efficient network simulator.
        Technical Report CMU-CS-99-164 (Pittsburgh, PA: Carnegie Mellon
        University, Department of Computer Science).
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

double adjust_target(double y, double d, double tr, double zr)
{
        /* 
         * Unit's activation is within zero error radius of the target, so
         * set its target to equal this activation.
         */
        if ((y - d < zr) && (y - d > -zr))
                return y;

        /*
         * Unit's activation is not within target radius of the target, so
         * adjust the unit's target towards the unit's activation by the
         * target radius.
         */

        /* adjust upward */
        if (y - d > tr)
                return d + tr;

        /* adjust downward */
        if (y - d < -tr)
                return d - tr;

        /*
         * Unit's activation is not within zero-error radius, but it is
         * within the target radius, so set the unit's target to equal this
         * activation.
         *
         * Note: This should only happen if the the zero-error radius is
         * zero, as a zero-error radius becomes meaningless if it is smaller
         * than the target radius.
         */
        return y;
}

                /************************
                 **** sum of squares ****
                 ************************/

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Sum squared error:

        se = 1/2 sum_i (y_i - d_i) ^ 2

and its derivative:

        se' = y_i - d_i
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

double err_fun_sum_of_squares(struct network *n, struct group *g,
        struct vector *t)
{
        double se = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:se) if (n->flags->omp_mthreaded)
#endif /* _OPENMP */
        for (uint32_t i = 0; i < g->vector->size; i++) {
                double y = g->vector->elements[i];
                double d = adjust_target(y, t->elements[i],
                        n->pars->target_radius, n->pars->zero_error_radius);
                
                se += pow(y - d, 2.0);
        }

        return 0.5 * se;
}

void err_fun_sum_of_squares_deriv(struct network *n, struct group *g,
        struct vector *t)
{
#ifdef _OPENMP
#pragma omp parallel for if (n->flags->omp_mthreaded)
#endif /* _OPENMP */
        for (uint32_t i = 0; i < g->vector->size; i++) {
                double y = g->vector->elements[i];
                double d = adjust_target(y, t->elements[i],
                        n->pars->target_radius, n->pars->zero_error_radius);
                
                g->error->elements[i] = y - d;
        }
}

                /***********************
                 **** cross entropy ****
                 ***********************/

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Cross entropy error:

        ce = sum_i log(d_i / y_i) * d_i + log((1 - d_i)
                / (1 - y_i)) * (1 - d_i)

and its derivative:
 
        ce' = (y_i - d_i) / (y_i * (1 - y_i))

Formulas and limit handling adapted from LENS (Rohde, 1999) source code.

Rohde, D. L. T. (1999). LENS: the light, efficient network simulator.
        Technical Report CMU-CS-99-164 (Pittsburgh, PA: Carnegie Mellon
        University, Department of Computer Science).

Also see:

Bishop, C. M. (1995). Neural Networks for Pattern Recognition. Oxford
        University Press: New York.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

double err_fun_cross_entropy(struct network *n, struct group *g,
        struct vector *t)
{
        double ce = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:ce) if (n->flags->omp_mthreaded)
#endif /* _OPENMP */
        for (uint32_t i = 0; i < g->vector->size; i++) {
                double y = g->vector->elements[i];
                double d = adjust_target(y, t->elements[i],
                        n->pars->target_radius, n->pars->zero_error_radius);
                
                if (d == 0.0) {
                        /*
                         * If d = 0 and y = 1, we obtain:
                         *
                         * log(0 / 1) * 0 + log((1 - 0) / (1 - 1)) * (1 - 0)
                         *      = -Inf * 0 + Inf * 1
                         *      = Inf
                         *
                         * We handle this by incrementing ce with LARGE_VALUE.
                         */
                        if (y == 1.0)
                                ce += LARGE_VALUE;

                        /*
                         * If, by contrast, d = 0 and y != 1, we obtain:
                         *
                         * log(0 / y) * 0 + log((1 - 0) / (1 - y)) * (1 - 0)
                         *      = 0 + log(1 / (1 - y)) * 1
                         *      = -log(1 - y)
                         */
                        else
                                ce += -log(1.0 - y);
                } else if (d == 1.0) {
                        /*
                         * If d = 1 and y = 0, we obtain:
                         *
                         * log(1 / 0) * 1 + log((1 - 1) / (1 - 0)) * (1 - 1)
                         *      = Inf * 1 + -Inf * 0
                         *      = Inf
                         *
                         * We handle this by incrementing ce with LARGE_VALUE.
                         */
                        if (y == 0.0)
                                ce += LARGE_VALUE;

                        /*
                         * If d = 1 and y != 0, we obtain:
                         *
                         * log(1 / y) * 1 + log((1 - 1) / (1 - y)) * (1 - 1)
                         *      = log(1 / y) * 1 + 0
                         *      = log(1 / y)
                         *      = -log(y)
                         */
                        else
                                ce += -log(y);
                } else {
                        /*
                         * if d != 0 and d != 1, and y <= 0 or y >= 1.0, we obtain:
                         *
                         * log(d / 0) * d + log((1 - d) / (1 - 0)) * (1 - d)
                         *      = Inf * d + log((1 - d) / (1 - 0)) * (1 - d)
                         *      = Inf
                         *
                         * or
                         *
                         * log(d / 1) * d + log((1 - d) / (1 - 1)) * (1 - d)
                         *      = log(d / 1) * d + Inf * (1 - d)
                         *      = Inf
                         *
                         * We handle this by incrementing ce with LARGE_VALUE.
                         */
                        if (y <= 0.0 || y >= 1.0)
                                ce += LARGE_VALUE;
                        /*
                         * Otherwise, simply increment ce by:
                         *
                         * log(d / y) * d + log((1 - d) / (1 - y)) * (1 - d)
                         */
                        else
                                ce += log(d / y)
                                        * d
                                        + log((1.0 - d) / (1.0 - y))
                                        * (1.0 - d);
                }
        }

        return ce;
}

void err_fun_cross_entropy_deriv(struct network *n, struct group *g,
        struct vector *t)
{
#ifdef _OPENMP
#pragma omp parallel for if (n->flags->omp_mthreaded)
#endif /* _OPENMP */
        for (uint32_t i = 0; i < g->vector->size; i++) {
                double y = g->vector->elements[i];
                double d = adjust_target(y, t->elements[i],
                        n->pars->target_radius, n->pars->zero_error_radius);
                
                if (d == 0.0) {
                        /*
                         * If d = 0 and 1 - y <= SMALL_VALUE, we obtain:
                         *
                         * (y - 0) / (y * (1 - y))
                         *      = y / (y * SMALL_VALUE)
                         *      = LARGE_VALUE
                         */
                        if (1.0 - y <= SMALL_VALUE)
                                g->error->elements[i] = LARGE_VALUE;
                        /*
                         * If d = 0 and 1 - y > SMALL_VALUE, we obtain:
                         *
                         * (y - 0) / (y * (1 - y))
                         *      = (y * 1) / (y * (1 - y))
                         *      = 1 / (1 - y)
                         */
                        else
                                g->error->elements[i] = 1.0 / (1.0 - y);
                } else if (d == 1.0) {
                        /*
                         * If d = 1 and y <= SMALL_VALUE, we obtain:
                         *
                         * (y - 1) / (y * (1 - y))
                         *      = (SMALL_VALUE - 1) /
                         *        (SMALL_VALUE * (1 - SMALL_VALUE))
                         *      = (-1 * (1 - SMALL_VALUE)) / 
                         *        (SMALL_VALUE * (1 - SMALL_VALUE))
                         *      = -1 / SMALL_VALUE
                         *      = -LARGE_VALUE
                         */
                        if (y <= SMALL_VALUE)
                                g->error->elements[i] = -LARGE_VALUE;
                        /*
                         * If d = 1 and y > SMALL_VALUE, we obtain:
                         *
                         * (y - 1) / (y * (1 - y))
                         *      = (-1 * (1 - y)) / (y * (1 - y))
                         *      = -1 / y
                         */
                        else
                                g->error->elements[i] = -1.0 / y;
                } else {
                        /*
                         * If d != 0 and d != 1, and y * (1 - y) <=
                         * SMALL_VALUE, we obtain:
                         *
                         * (y - d) / SMALL_VALUE
                         *      = (y = d) * LARGE_VALUE
                         */
                        if (y * (1.0 - y) <= SMALL_VALUE)
                                g->error->elements[i] = (y - d) * LARGE_VALUE;
                        /*
                         * Otherwise, we simply compute:
                         *
                         * (y - d) / (y * (1 - y))
                         */
                        else
                                g->error->elements[i] = (y - d) / (y * (1.0 - y));
                }
        }
}

                /********************
                 **** divergence ****
                 ********************/

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Divergence error:

        de = sum_i log(d_i / y_i) * d_i

and its derivative:

        de' = -d_i / y_i

Formulas and limit handling adapted from LENS (Rohde, 1999) source code.

Rohde, D. L. T. (1999). LENS: the light, efficient network simulator.
        Technical Report CMU-CS-99-164 (Pittsburgh, PA: Carnegie Mellon
        University, Department of Computer Science).
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

double err_fun_divergence(struct network *n, struct group *g,
        struct vector *t)
{
        double de = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:de) if (n->flags->omp_mthreaded)
#endif /* _OPENMP */
        for (uint32_t i = 0; i < g->vector->size; i++) {
                double y = g->vector->elements[i];
                double d = adjust_target(y, t->elements[i],
                        n->pars->target_radius, n->pars->zero_error_radius);

                /*
                 * If d is 0, we obtain:
                 * 
                 * log(0.0 / y) * 0 = -Inf * 0.0
                 *
                 * We handle this by incrementing de by 0.
                 */
                if (d == 0.0) {
                        de += 0.0;
                /*
                 * If y <= SMALL_VALUE, we obtain:
                 *
                 * log(d / SMALL_VALUE) * d
                 *      = log(d * LARGE_VALUE) * d
                 */
                } else if (y <= SMALL_VALUE) {
                        de += d * log(d * LARGE_VALUE);

                /*
                 * Otherwise, we simply increment de by:
                 *
                 * log(d / y) * d
                 */
                } else {
                        de += log (d / y) * d;
                }
        }

        return de;
}

void err_fun_divergence_deriv(struct network *n, struct group *g,
        struct vector *t)
{
#ifdef _OPENMP
#pragma omp parallel for if (n->flags->omp_mthreaded)
#endif /* _OPENMP */
        for (uint32_t i = 0; i < g->vector->size; i++) {
                double y = g->vector->elements[i];
                double d = adjust_target(y, t->elements[i],
                        n->pars->target_radius, n->pars->zero_error_radius);
                
                /*
                 * If d = 0, we obtain:
                 *
                 * -0.0 / y = 0
                 *
                 * So, we simply set the error to 0.
                 */
                if (d == 0) {
                        g->error->elements[i] = 0.0;

                /*
                 * If y <= SMALL_VALUE, we obtain:
                 *
                 * -d / y 
                 *      = -d / SMALL_VALUE
                 *      = -d * LARGE_VALUE
                 */
                } else if (y <= SMALL_VALUE) {
                        g->error->elements[i] = -d * LARGE_VALUE;

                /*
                 * Otherwise, we simply compute:
                 *
                 * -d / y
                 */
                } else {
                        g->error->elements[i] = -d / y;
                }
        }
}
