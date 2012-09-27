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
#include "pprint.h"

#include <math.h>

void compute_erp_correlates(struct network *n)
{
        mprintf("computing ERP correlates for network: [%s]", n->name);

        /* find "Wernicke" and "Broca" */
        struct group *w, *b;
        if (!(w = find_group_by_name(n, "hidden")))
                goto error_out;
        if (!(b = find_group_by_name(n, "output")))
                goto error_out;

        struct vector *pw = create_vector(w->vector->size);
        struct vector *pb = create_vector(b->vector->size);

        /* present test set to network */
        for (int i = 0; i < n->test_set->num_elements; i++) {
                struct element *e = n->test_set->elements[i];

                /* reset context groups */
                if (n->srn)
                        reset_context_groups(n);

                zero_out_vector(pb);
                zero_out_vector(pw);

                rprintf("\n\nI: \"%s\"", e->name);
                char *tokens = strtok(e->name, " ");

                for (int j = 0; j < e->num_events; j++) {
                        copy_vector(n->input->vector, e->inputs[j]);
                        feed_forward(n, n->input);

                        double n400_correlate = compute_n400_correlate(w->vector, pw);
                        double p600_correlate = compute_p600_correlate(b->vector, pb);

                        /*
                        printf("%s\n", tokens);
                        print_vector(pw);
                        print_vector(w->vector);

                        printf("\n");
                        print_vector(pb);
                        print_vector(b->vector);
                        */

                        printf("\n%s\t\tN400: %f\t\tP600: %f\n", tokens, n400_correlate, p600_correlate);

                        pprint_vector(pw);
                        pprint_vector(w->vector);

                        if(e->targets[j]) {
                                printf("\n");
                                printf("T: ");
                                pprint_vector(e->targets[j]);
                                printf("O: ");
                                pprint_vector(n->output->vector);
                                printf("\n\n");
                        }

                        copy_vector(pw, w->vector);
                        copy_vector(pb, b->vector);

                        tokens = strtok(NULL, " ");
                }
        }
        
        return;

error_out:
        perror("[compute_erp_correlates()]");
        return;
}

/*
double compute_n400_correlate(struct vector *v, struct vector *pv)
{
        double sa = 0.0;
        for (int i = 0; i < v->size; i++)
                sa += v->elements[i] + 1;

        return sa / v->size;
}
*/

double compute_n400_correlate(struct vector *v, struct vector *pv)
{
        double ab = 0.0;
        for (int i = 0; i < v->size; i++)
                ab += pow(v->elements[i] - pv->elements[i], 2.0);

        return sqrt(ab);
}

double compute_p600_correlate(struct vector *v, struct vector *pv)
{
        double ab = 0.0;
        for (int i = 0; i < v->size; i++)
                ab += pow(v->elements[i] - pv->elements[i], 2.0);

        return sqrt(ab);
}
