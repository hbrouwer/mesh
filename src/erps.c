/*
 * erps.c
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

#include "act.h"
#include "engine.h"
#include "erps.h"
#include "math.h"
#include "pprint.h"

#include <math.h>

void compute_erp_correlates(struct network *n)
{
        mprintf("computing ERP correlates for network: [%s]", n->name);

        /* find "Wernicke" and "Broca" */
        struct group *w;
        if (!(w = find_group_by_name(n, "hidden")))
                goto error_out;

        struct vector *pw = create_vector(w->vector->size);
        // struct vector *pb = create_vector(b->vector->size);

        /* present test set to network */
        for (int i = 0; i < n->test_set->num_elements; i++) {
                struct element *e = n->test_set->elements[i];

                /* reset context groups */
                if (n->type == TYPE_SRN)
                        reset_context_groups(n);

                zero_out_vector(pw);

                rprintf("\n\nI: \"%s\"", e->name);
                char *tokens = strtok(e->name, " ");

                for (int j = 0; j < e->num_events; j++) {
                        copy_vector(n->input->vector, e->inputs[j]);
                        feed_forward(n, n->input);

                        if (j > 0) {
                                double n400_correlate = compute_n400_correlate(w->vector, pw);
                                printf("N400: %f\n", n400_correlate);
                                pprint_vector(w->vector);
                                pprint_vector(pw);
                        }

                        copy_vector(pw, w->vector);

                        tokens = strtok(NULL, " ");
                }
        }
        
        return;

error_out:
        perror("[compute_erp_correlates()]");
        return;
}

double compute_n400_correlate(struct vector *v, struct vector *pv)
{
        double v_mean = 0.0, pv_mean = 0.0;
        for (int i = 0; i < v->size; i++) {
                v_mean += v->elements[i];
                pv_mean += pv->elements[i];
        }
        v_mean /= v->size;
        pv_mean /= pv->size;

        double nom = 0.0, denom = 0.0;
        for (int i = 0; i < v->size; i++) {
                nom += (v->elements[i] - v_mean) * (pv->elements[i] - pv_mean);
                denom += pow(v->elements[i] - v_mean, 2.0) * pow(pv->elements[i] - pv_mean, 2.0);
        }

        return 1.0 / (nom / pow(denom, 0.5));
}

double compute_p600_correlate(struct vector *v, struct vector *pv)
{
        double ab = 0.0;
        for (int i = 0; i < v->size; i++)
                ab += square(v->elements[i] - pv->elements[i]);

        return sqrt(ab);
}
