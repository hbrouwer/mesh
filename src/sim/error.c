/*
 * error.c
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

#include "error.h"

#include <math.h>

/*
 * ########################################################################
 * ## Sum of squares error                                               ##
 * ########################################################################
 */

double error_sum_of_squares(struct vector *o, struct vector *t)
{
        double se = 0.0;

        for (int i = 0; i < o->size; i++) {
                double y = o->elements[i];
                double d = t->elements[i];

                se += pow(y - d, 2.0);
        }

        return 0.5 * se;
}

struct vector *error_sum_of_squares_deriv(struct vector *o, struct vector *t)
{
        struct vector *e = create_vector(o->size);

        for (int i = 0; i < o->size; i++) {
                double y = o->elements[i];
                double d = t->elements[i];

                e->elements[i] = y - d;
        }

        return e;
}

/*
 * ########################################################################
 * ## Cross entropy error                                                ##
 * ########################################################################
 */

/*
 * Formulas and limit handling adapted from LENS source code.
 */

#define LARGE_VALUE 1e10
#define SMALL_VALUE 1e-10

double error_cross_entropy(struct vector *o, struct vector *t)
{
        double ce = 0.0;

        for (int i = 0; i < o->size; i++) {
                double y = o->elements[i];
                double d = t->elements[i];

                if (d == 0.0) {
                        if (y == 1.0)
                                ce += LARGE_VALUE;
                        else
                                ce += -log(1.0 - y);
                } else if (d == 1.0) {
                        if (y == 0.0)
                                ce += LARGE_VALUE;
                        else
                                ce += -log(y);
                } else {
                        if (y <= 0.0 || y >= 1.0)
                                ce += LARGE_VALUE;
                        else
                                ce += log(d / y)
                                        * d
                                        + log((1.0 - d) / (1.0 - y))
                                        * (1.0 - d);
                }
        }

        return ce;
}

struct vector *error_cross_entropy_deriv(struct vector *o, struct vector *t)
{
        struct vector *e = create_vector(o->size);

        for (int i = 0; i < o->size; i++) {
                double y = o->elements[i];
                double d = t->elements[i];

                if (d == 0.0) {
                        if (1.0 - y <= SMALL_VALUE)
                                e->elements[i] = LARGE_VALUE;
                        else
                                e->elements[i] = 1.0 / (1.0 - y);
                } else if (d == 1) {
                        if (y <= SMALL_VALUE)
                                e->elements[i] = -LARGE_VALUE;
                        else
                                e->elements[i] = -1.0 / y;
                } else {
                        if (y * (1.0 - y) <= SMALL_VALUE)
                                e->elements[i] = (y - d) * LARGE_VALUE;
                        else
                                e->elements[i] = (y - d) / (y * (1.0 - d));
                }
        }

        return e;
}

/*
 * ########################################################################
 * ## Divergence error                                                   ##
 * ########################################################################
 */

/*
 * Formulas and limit handling adapted from LENS source code.
 */

double error_divergence(struct vector *o, struct vector *t)
{
        double de = 0.0;

        for (int i = 0; i < o->size; i++) {
                double y = o->elements[i];
                double d = t->elements[i];

                if (d == 0.0) {
                        de += 0.0;
                } else if (y <= 0.0) {
                        de += d * log(d) * LARGE_VALUE;
                } else {
                        de += log (d / y) * d;
                }
        }

        return de;
}

struct vector *error_divergence_deriv(struct vector *o, struct vector *t)
{
        struct vector *e = create_vector(o->size);

        for (int i = 0; i < o->size; i++) {
                double y = o->elements[i];
                double d = t->elements[i];

                if (d == 0) {
                        e->elements[i] = 0.0;
                } else if (y <= SMALL_VALUE) {
                        e->elements[i] = -d * LARGE_VALUE;
                } else {
                        e->elements[i] = -d / y;
                }
        }

        return e;
}
