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

#include "erps.h"
#include "train.h"

#include <math.h>

void compute_erp_correlates(struct network *n)
{
        mprintf("computing ERP correlates for network: [%s]", n->name);

        struct vector *previous_output = create_vector(n->output->vector->size);
        for (int i = 0; i < n->test_set->num_elements; i++) {
                struct element *e = n->test_set->elements[i];

                rprintf("computing ERP correlates for item: %d -- \"%s\"", 
                                i, e->name);

                for (int j = 0; j < e->num_events; j++) {
                        copy_vector(n->input->vector, e->inputs[j]);
                        if (e->targets[j] != NULL)
                                copy_vector(n->target, e->targets[j]);
               
                        feed_forward(n, n->input);

                        // print_vector(n->output->vector);

                        if (j == e->num_events - 2) {
                                copy_vector(previous_output, n->output->vector);
                        }
                        
                        if (j == e->num_events - 1) {
                                double p600_correlate = compute_p600_correlate(
                                                n->output->vector, previous_output);
                                print_vector(previous_output);
                                print_vector(n->output->vector);
                                printf("P600-amplitude: %f\n", p600_correlate);
                        }
                }

                if (n->srn)
                        reset_elman_groups(n);
        }
}

double compute_p600_correlate(struct vector *o, struct vector *po)
{
        double ab = 0.0, a = 0.0, b = 0.0;
        for (int i = 0; i < o->size; i++) {
                ab += pow(o->elements[i] - po->elements[i], 2.0);
                //a += sqrt(pow(o->elements[i],2.0));
                //b += sqrt(pow(po->elements[i],2.0));
        }

        return sqrt(ab);
        
}

/*
double compute_p600_correlate(struct vector *o, struct vector *po)
{
        double ab = 0.0, a = 0.0, b = 0.0;
        for (int i = 0; i < o->size; i++) {
                ab += o->elements[i] * po->elements[i];
                a += sqrt(pow(o->elements[i],2.0));
                b += sqrt(pow(po->elements[i],2.0));
        }

        return ab / (a * b);
}
*/
