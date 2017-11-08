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

#include <math.h>
#include <string.h>

#include "dsys.h"

#include "../act.h"
#include "../main.h"
#include "../math.h"

                /*************************
                 **** dynamic systems ****
                 *************************/

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
This implements machinery to transform a connectionist model into a dynamic
system by turning the activation function of a specified group:

        a_out(i+1) = f(W_out a_rec(i+1) + b_out)

into a simple differential equation:

        da_out
        ------ = f(W_out a_rec(i+1) + b_out) - a_out
          dt

such that a_out changes from a_out(i) into a_out(i+1) over processing time
(cf. Frank & Viliocco, 2011). Ideally, this process converges when da_out/dt
= 0, meaning that a_out(i) = aout(i+1) = f(W_out a_rec(i+1) + b_out). However, 
as convergence is aymptotic, this will never happen, and as such the process
is stopped when:

        |da_out/dt| < max{1.0 * |a_out|, 10^-8}

References

Frank, S. L. and Vigliocco, G. (2011). Sentence comprehension as mental
        simulation: an information-theoretic perspective. Information, 2,
        672-696.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

void dsys_proc_time(struct network *n, struct group *g, struct item *item)
{
        struct vector *pv = create_vector(g->vector->size);
        fill_vector_with_value(pv, 1.0);
        fill_vector_with_value(pv, 1.0 / euclidean_norm(pv));

        size_t block_size = strlen(item->name) + 1;
        char sentence[block_size];
        memset(&sentence, 0, block_size);
        strncpy(sentence, item->name, block_size - 1);

        cprintf("\n");

        uint32_t col_len = 10;

        /* print the words of the sentence */
        for (uint32_t i = 0; i < col_len; i++)
                cprintf(" ");
        char *token = strtok(sentence, " ");
        do {
                cprintf("\x1b[35m%s\x1b[0m", token);
                for (uint32_t i = 0; i < col_len - strlen(token); i++)
                        cprintf(" ");
                token = strtok(NULL, " ");
        } while (token);
        cprintf("\n\n");

        cprintf("ProcTime: ");
        if (n->type == ntype_srn)
                reset_context_groups(n);
        for (uint32_t i = 0; i < item->num_events; i++) {
                /* feed activation forward */
                if (i > 0 && n->type == ntype_srn)
                        shift_context_groups(n);
                copy_vector(n->input->vector, item->inputs[i]);
                feed_forward(n, n->input);

                double t = dsys_compute_proc_time(n, pv, g->vector);
                
                cprintf("%.5f", t);
                for (uint32_t i = 0; i < col_len - 7; i++)
                        cprintf(" ");

                copy_vector(pv, g->vector);
        }
        cprintf("\n\n");

        free_vector(pv);

        return;
}

double dsys_compute_proc_time(struct network *n, struct vector *a_out0,
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
                                &dsys_unit_act,
                                h, a_out1->elements[i],
                                a_outx->elements[i]);

                /* update dt */
                dt += h;

                /* compute da_out/dt */
                for (uint32_t i = 0; i < a_out0->size; i++)
                        da_out_dt->elements[i] =
                                (a_outx->elements[i] - a_out0->elements[i]) / dt;

                /* compute norm for a_out */
                norm_a_outx = euclidean_norm(a_outx);

                /* compute norm for da_out/dt */
                norm_da_out_dt = euclidean_norm(da_out_dt);

        } while (norm_da_out_dt > maximum(0.1 * norm_a_outx, pow(10.0, -8.0)));

        free_vector(da_out_dt);
        free_vector(a_outx);

        return dt;
}

double dsys_unit_act(double yn1, double yn0)
{
        return yn1 - yn0;
}
