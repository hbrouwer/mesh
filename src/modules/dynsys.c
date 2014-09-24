/*
 * dynsys.c
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

#include <math.h>

#include "dynsys.h"

#include "../act.h"
#include "../math.h"

/**************************************************************************
 *************************************************************************/
void dynsys_test_item(struct network *n, struct group *g, struct item *item)
{
        if (n->type == TYPE_SRN)
                reset_context_groups(n);

        /* initial vector */
        struct vector *pv = create_vector(g->vector->size);
        fill_vector_with_value(pv, 0.5);

        for (uint32_t i = 0; i < item->num_events; i++) {
                /*
                 * Shift context group chain, in case of 
                 * "Elman-towers".
                 */
                if (i > 0 && n->type == TYPE_SRN)
                        shift_context_groups(n);

                copy_vector(n->input->vector, item->inputs[i]);
                feed_forward(n, n->input);

                double t = dynsys_processing_time(n, pv, g->vector);
                printf("Processing time for event %d: %f\n", i, t);

                copy_vector(pv, g->vector);
        }

        dispose_vector(pv);

        return;
}

/**************************************************************************
 *************************************************************************/

/**************************************************************************
 *************************************************************************/
double dynsys_processing_time(struct network *n, struct vector *iv,
                struct vector *tv)
{
        struct vector *diff = create_vector(tv->size);
        struct vector *step = create_vector(tv->size);
        struct vector *delt = create_vector(tv->size);
        copy_vector(diff, iv);

        ///////////////////////////////////

        double h = 0.01;
        double t = 0.0;

        double delt_en, step_en;
        do {
                copy_vector(step, diff);

                for (uint32_t i = 0; i < tv->size; i++) {
                        diff->elements[i] = runge_kutta4(&dynsys_unit_act, h, tv->elements[i], step->elements[i]);
                        delt->elements[i] = (diff->elements[i] - step->elements[i]) / t;
                }

                t += h;

                print_vector(diff);

                delt_en = euclidean_norm(delt);
                step_en = euclidean_norm(step);

        } while (delt_en > maximum(0.1 * step_en, pow(10.0,-8.0)));

        //////////////////////////////////

        dispose_vector(diff);
        dispose_vector(step);
        dispose_vector(delt);

        return t;
}

/**************************************************************************
 *************************************************************************/
double dynsys_unit_act(double yn1, double yn0)
{
        return yn1 - yn0;
}
