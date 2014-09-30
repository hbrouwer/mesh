/*
 * dss.c
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

#include "dss.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../act.h"
#include "../main.h"
#include "../math.h"

/**************************************************************************
 * This implements a variety of functions for dealing with Distributed
 * Situation Space vectors, see:
 *
 * Frank, S. L., Haselager, W. F. G, & van Rooij, I. (2009). Connectionist
 *     semantic systematicity. Cognition, 110, 358-379.
 *************************************************************************/

/**************************************************************************
 *************************************************************************/
void dss_test(struct network *n)
{
        double acs = 0.0;
        uint32_t ncs = 0; 

        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                struct item *item = n->asp->items->elements[i];

                if (n->type == TYPE_SRN)
                        reset_context_groups(n);

                for (uint32_t j = 0; j < item->num_events; j++) {
                        /*
                         * Shift context group chain, in case of 
                         * "Elman-towers".
                         */
                        if (j > 0 && n->type == TYPE_SRN)
                                shift_context_groups(n);

                        copy_vector(n->input->vector, item->inputs[j]);
                        feed_forward(n, n->input);
                }

                double tau = dss_comprehension_score(
                                item->targets[item->num_events - 1],
                                n->output->vector);


                if (tau > 0.0)
                        pprintf("\x1b[32m%s: %f\x1b[0m\n", item->name, tau);
                else
                        pprintf("\x1b[31m%s: %f\x1b[0m\n", item->name, tau);

                if (!isnan(tau)) {
                        acs += tau;
                        ncs++;
                }
        }

        pprintf("\nAverage comprehension score: (%f / %d =) %f\n",
                        acs, ncs, acs / ncs);
}

/**************************************************************************
 *************************************************************************/
void dss_test_item(struct network *n, struct item *item)
{
        if (n->type == TYPE_SRN)
                reset_context_groups(n);

        for (uint32_t i = 0; i < item->num_events; i++) {
                /*
                 * Shift context group chain, in case of 
                 * "Elman-towers".
                 */
                if (i > 0 && n->type == TYPE_SRN)
                        shift_context_groups(n);

                copy_vector(n->input->vector, item->inputs[i]);
                feed_forward(n, n->input);

                double cs = dss_comprehension_score(item->targets[i],
                                n->output->vector);
                printf("event %d -- comprehension score: %f\n", i, cs);
        }
}

/**************************************************************************
 *************************************************************************/
void dss_beliefs(struct network *n, struct set *set, struct item *item)
{
        if (n->type == TYPE_SRN)
                reset_context_groups(n);

        for (uint32_t i = 0; i < item->num_events; i++) {
                /*
                 * Shift context group chain, in case of 
                 * "Elman-towers".
                 */
                if (i > 0 && n->type == TYPE_SRN)
                        shift_context_groups(n);

                copy_vector(n->input->vector, item->inputs[i]);
                feed_forward(n, n->input);
        }

        double cs = dss_comprehension_score(item->targets[item->num_events - 1],
                        n->output->vector);
        pprintf("\nsemantics: %s\n", item->meta);
        pprintf("comprehension score: %f\n\n", cs);

        for (uint32_t i = 0; i < set->items->num_elements; i++) {
                struct item *probe = set->items->elements[i];

                double tau = dss_comprehension_score(
                                probe->targets[probe->num_events - 1],
                                n->output->vector);

                if (tau > 0.0)
                        pprintf("\x1b[32m%s: %f\x1b[0m\n", probe->name, tau);
                else
                        pprintf("\x1b[31m%s: %f\x1b[0m\n", probe->name, tau);
        }
        printf("\n");
}

/**************************************************************************
 * This computes the comprehension score (Frank et al., 2009), which is 
 * defined as:
 *
 *                     | tau(a|z) - tau(a)
 *                     | ----------------- , if tau(a|z) > tau(a)
 *                     |    1 - tau(a)
 *     comprehension = |
 *                     | tau(a|z) - tau(a)
 *                     | ----------------- , otherwise
 *                             tau(a)
 *
 * where tau(a|z) is the conditional belief of a given z, and tau(a) is the
 * prior belief in a.
 *
 * If tau(a|z) = 1, the comprehension score is maximal: +1. On the other
 * hand, if tau(a|z) = 0, the comprehension score is minimal: -1.
 * Intuitively, a positive comprehension score is a measure of how much
 * uncertainty in event a is taken away by z, whereas a negative
 * comprehension score measures how much certainty in event a is taken away
 * by z.
 *
 * References
 *
 * Frank, S. L., Haselager, W. F. G, & van Rooij, I. (2009). Connectionist
 *     semantic systematicity. Cognition, 110, 358-379.
 *************************************************************************/
double dss_comprehension_score(struct vector *a, struct vector *z)
{
        double cs = 0.0;

        double tau_a_given_z = dss_tau_conditional(a, z);
        double tau_a = dss_tau_prior(a);

        /* unlawful event */
        if (tau_a == 0.0)
                return NAN;

        if (tau_a_given_z > tau_a)
                cs = (tau_a_given_z - tau_a) / (1.0 - tau_a);
        else
                cs = (tau_a_given_z - tau_a) / tau_a;

        return cs;
}

/**************************************************************************
 * Prior belief in a:
 *
 * tau(a) = 1/n sum_i u_i(a)
 *************************************************************************/
double dss_tau_prior(struct vector *a)
{
        double tau = 0.0;

        for (uint32_t i = 0; i < a->size; i++)
                tau += a->elements[i];

        return tau / a->size;
}

/**************************************************************************
 * Belief the conjunction of a and b:
 *
 * tau(a^b) = 1/n sum_i u_i(a) * u_i(b)
 *************************************************************************/
double dss_tau_conjunction(struct vector *a, struct vector *b)
{
        double tau = 0.0;

        if (a == b)
                tau = dss_tau_prior(a);
        else
                tau = inner_product(a, b) / a->size;
        
        return tau;
}

/**************************************************************************
 * Conditional belief in a given b:
 *
 * tau(a|b) = tau(a^b) / tau(b)
 *************************************************************************/
double dss_tau_conditional(struct vector *a, struct vector *b)
{
        return dss_tau_conjunction(a, b) / dss_tau_prior(b);
}

/**************************************************************************
 * Word information metrics, as described in:
 *
 * Frank, S. L. and Vigliocco, G. (2011). Sentence comprehension as mental
 *     simulation: an information-theoretic perspective. Information, 2,
 *     672-696.
 *************************************************************************/
void dss_word_information(struct network *n, struct item *item)
{
        char prefix1[strlen(item->name) + 1];
        strncpy(prefix1, item->name, strlen(item->name));
        char prefix2[strlen(item->name) + 1];
        strncpy(prefix2, item->name, strlen(item->name));

        /* frequency table */
        int32_t freq_table[n->asp->items->num_elements];
        memset(&freq_table, 0, n->asp->items->num_elements * sizeof(int32_t));
        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                /* skip seen items */
                if (freq_table[i] == -1) 
                        continue;
                struct item *item1 = n->asp->items->elements[i];
                for (uint32_t j = 0; j < n->asp->items->num_elements; j++) {
                        struct item *item2 = n->asp->items->elements[j];
                        if (strcmp(item1->name, item2->name) == 0) {
                                freq_table[i]++;
                                if (i != j)
                                        freq_table[j] = -1; /* mark as seen */
                        }
                }
        }

        pprintf("Word    \tSsyn    \tD_Hsyn  \tSsem    \tD_Hsem  \n");
        pprintf("========\t========\t========\t========\t========\n");

        if (n->type == TYPE_SRN)
                reset_context_groups(n);

        for (uint32_t i = 0; i < item->num_events; i++) {
                /* isolate prefixes */
                uint32_t wp = 0, cp = 0;
                uint32_t i1 = 0, i2 = strlen(item->name);
                for (char *p = item->name; *p != '\0'; p++) {
                        if (*p == ' ') {
                                wp++;
                                if (wp == i)
                                        i1 = cp;
                                if (wp == i + 1)
                                        i2 = cp;
                        }
                        cp++;
                }
                char c1 = prefix1[i1];
                char c2 = prefix2[i2];
                prefix1[i1] = '\0';
                prefix2[i2] = '\0';

                /* compute prefix frequencies */
                uint32_t freq_prefix1 = 0.0, freq_prefix2 = 0.0;
                for (uint32_t j = 0; j < n->asp->items->num_elements; j++) {
                        struct item *titem = n->asp->items->elements[j];
                        if (strncmp(titem->name, prefix1, strlen(prefix1)) == 0)
                                freq_prefix1++;
                        if (strncmp(titem->name, prefix2, strlen(prefix2)) == 0)
                                freq_prefix2++;
                }

                /* compute syntactic entropies */
                double hsyn1 = 0.0, hsyn2 = 0.0;
                for (uint32_t j = 0; j < n->asp->items->num_elements; j++) {
                        double freq = freq_table[j];
                        if (freq < 0.0)
                                continue; /* skip doubles */
                        struct item *titem = n->asp->items->elements[j];
                        if (strncmp(titem->name, prefix1, strlen(prefix1)) == 0)
                                hsyn1 += (freq / freq_prefix1) * log(freq / freq_prefix1);
                        if (strncmp(titem->name, prefix2, strlen(prefix2)) == 0)
                                hsyn2 += (freq / freq_prefix2) * log(freq / freq_prefix2);
                }
                hsyn1 = -hsyn1;
                hsyn2 = -hsyn2;

                /* restore prefixes */
                prefix1[i1] = c1;
                prefix2[i2] = c2;

                /*
                 * Shift context group chain, in case of 
                 * "Elman-towers".
                 */
                if (i > 0 && n->type == TYPE_SRN)
                        shift_context_groups(n);

                copy_vector(n->input->vector, item->inputs[i]);
                feed_forward(n, n->input);

                /* compute metrics */
                double ssyn = log(freq_prefix1) - log(freq_prefix2);
                double ssem = 0.0; // -log(dss_tau_conditional(n->output->vector, pv));
//                double ssem2 = log(dss_tau_prior(pv)) - log(dss_tau_prior(n->output->vector));
                double delta_hsyn = hsyn1 - hsyn2;

                pprintf("");
                if (i1 > 0) i1++;
                for (uint32_t j = i1; j < i2; j++) {
                        putchar(item->name[j]);
                }
                if (i2 - i1 < 3) printf("\t");
                printf("\t%f\t%f\t%f\n", ssyn, delta_hsyn, ssem);
        }

        return;
}
