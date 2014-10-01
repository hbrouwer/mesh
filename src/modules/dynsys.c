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
 * This implements machinery to transform a connectionist model into a 
 * dynamic system by turning the activation function of a specified group:
 *
 *     a_out(i+1) = f(W_out a_rec(i+1) + b_out)
 *
 * into a simple differential equation:
 *
 *     da_out
 *     ------ = f(W_out a_rec(i+1) + b_out) - a_out
 *       dt
 *
 * such that a_out changes from a_out(i) into a_out(i+1) over processing
 * time (cf. Frank & Viliocco, 2011). Ideally, this process converges when
 * da_out/dt = 0, meaning that a_out(i) = aout(i+1) = f(W_out a_rec(i+1)
 * + b_out). However, as convergence is aymptotic, this will never happen,
 * and as such the process is stopped when:
 *
 *     |da_out/dt| < max{1.0 * |a_out|, 10^-8}
 *
 * References
 *
 * Frank, S. L. and Vigliocco, G. (2011). Sentence comprehension as mental
 *     simulation: an information-theoretic perspective. Information, 2,
 *     672-696.
 *************************************************************************/

/**************************************************************************
 *************************************************************************/
void dynsys_test_item(struct network *n, struct group *g, struct item *item)
{
        /* XXX: initial vector should be the unit vector */
        struct vector *pv = create_vector(g->vector->size);

        if (n->type == TYPE_SRN)
                reset_context_groups(n);
        for (uint32_t i = 0; i < item->num_events; i++) {
                /* feed activation forward */
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
double dynsys_processing_time(struct network *n, struct vector *a_out0,
                struct vector *a_out1)
{
        struct vector *da_out_dt = create_vector(a_out0->size);
        struct vector *a_outx = create_vector(a_out0->size);
        copy_vector(a_outx, a_out0);

        double h = 0.001; /* step size */
        double dt = 0.0;  /* time */

        double norm_da_out_dt = 0.0; 
        double norm_a_outx = 0.0;

        do {
                /* update a_out */
                for (uint32_t i = 0; i < a_out0->size; i++)
                        a_outx->elements[i] = runge_kutta4(
                                        &dynsys_unit_act, h,
                                        a_out1->elements[i],
                                        a_outx->elements[i]);

                /* update dt */
                dt += h;

                /* compute da_out/dt */
                for (uint32_t i = 0; i < a_out0->size; i++)
                        da_out_dt->elements[i] = (a_outx->elements[i]
                                        - a_out0->elements[i]) / dt;

                /* compute norm for a_out */
                norm_a_outx = euclidean_norm(a_outx);

                /* compute norm for da_out/dt */
                norm_da_out_dt = euclidean_norm(da_out_dt);

        } while (norm_da_out_dt > maximum(0.1 * norm_a_outx, pow(10.0, -8.0)));

        dispose_vector(da_out_dt);
        dispose_vector(a_outx);

        return dt;
}

/**************************************************************************
 *************************************************************************/
double dynsys_unit_act(double yn1, double yn0)
{
        return yn1 - yn0;
}
