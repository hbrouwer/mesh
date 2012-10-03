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

#include <float.h>
#include <math.h>

/*
 * ########################################################################
 * ## Sum of squares error                                               ##
 * ########################################################################
 */

double error_sum_of_squares(struct network *n)
{
        double se = 0.0;

        for (int i = 0; i < n->output->vector->size; i++) {
                double y = n->output->vector->elements[i];
                double d = n->target->elements[i];

                se += pow(y - d, 2.0);
        }

        return 0.5 * se;
}

struct vector *error_sum_of_squares_deriv(struct network *n)
{
        struct vector *e = create_vector(n->target->size);

        for (int i = 0; i < n->output->vector->size; i++) {
                double y = n->output->vector->elements[i];
                double d = n->target->elements[i];

                e->elements[i] = y - d;
        }

        return e;
}

/*
 * ########################################################################
 * ## Cross entropy error                                                ##
 * ########################################################################
 */

/* XXX: no idea if this handles the +Inf and -Inf limits correctly */

double error_cross_entropy(struct network *n)
{
        double ce = 0.0;

        for (int i = 0; i < n->output->vector->size; i++) {
                double y = n->output->vector->elements[i];
                double d = n->target->elements[i];

                /* target is zero */
                if (d == 0.0) {
                        if (y == 1.0)
                                ce += DBL_MAX;
                        else
                                ce += -log(1.0 - y);

                /* target is one */
                } else if (d == 1.0) {
                        if (y == 0.0)
                                ce += DBL_MAX;
                        else
                                ce += -log(y);
               
                /* otherwise */
                } else {
                        ce += log(d / y) * d
                                + log((1.0 - d) / (1.0 - y))
                                * (1.0 - d);
                }
        }

        return ce;
}

struct vector *error_cross_entropy_deriv(struct network *n)
{
        struct vector *e = create_vector(n->target->size);

        for (int i = 0; i < n->output->vector->size; i++) {
                double y = n->output->vector->elements[i];
                double d = n->target->elements[i];
                               
                e->elements[i] = y - d;
        }

        return e;
}
