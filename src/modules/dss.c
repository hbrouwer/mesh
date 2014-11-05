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
                        /* feed activation forward */
                        if (j > 0 && n->type == TYPE_SRN)
                                shift_context_groups(n);
                        copy_vector(n->input->vector, item->inputs[j]);
                        feed_forward(n, n->input);
                }

                struct vector *target = item->targets[item->num_events - 1];
                double tau = dss_comprehension_score(target, n->output->vector);

                tau > 0.0 ? pprintf("\x1b[32m%s: %f\x1b[0m\n", item->name, tau)
                        : pprintf("\x1b[31m%s: %f\x1b[0m\n", item->name, tau);

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
void dss_beliefs(struct network *n, struct set *set, struct item *item)
{
        struct matrix *taus = create_matrix(set->items->num_elements,
                        item->num_events);

        /* compute belief values */
        if (n->type == TYPE_SRN)
                reset_context_groups(n);
        for (uint32_t i = 0; i < item->num_events; i++) {
                /* feed activation forward */
                if (i > 0 && n->type == TYPE_SRN)
                        shift_context_groups(n);
                copy_vector(n->input->vector, item->inputs[i]);
                feed_forward(n, n->input);

                for (uint32_t j = 0; j < set->items->num_elements; j++) {
                        struct item *probe = set->items->elements[j];
                        taus->elements[j][i] = dss_comprehension_score(
                                        probe->targets[probe->num_events - 1],
                                        n->output->vector);
                }
        }

        /* compute comprehension score */
        struct vector *target = item->targets[item->num_events - 1];
        double cs = dss_comprehension_score(target, n->output->vector);
        pprintf("Sentence: %s\n", item->name);
        pprintf("Semantics: %s\n", item->meta);
        pprintf("Comprehension score: %f\n", cs);
        pprintf("\n");

        /* determine longest basic event name */
        uint32_t max_len = 0;
        for (uint32_t i = 0; i < set->items->num_elements; i++) {
                struct item *probe = set->items->elements[i];
                uint32_t len = strlen(probe->name);
                if (len > max_len)
                        max_len = len;
        }
        max_len++;

        uint32_t col_len = 20;

        char sentence[strlen(item->name) + 1];
        memset(&sentence, 0, strlen(item->name) + 1);
        strncpy(sentence, item->name, strlen(item->name));
        pprintf("");
        for (uint32_t i = 0; i < max_len; i++)
                printf(" ");
        char *token = strtok(sentence, " ");
        do {
                printf("%s", token);
                uint32_t whitespace = col_len - strlen(token);
                for (uint32_t j = 0; j < whitespace; j++)
                        printf(" ");
                token = strtok(NULL, " ");
        } while (token);
        printf("\n");
        pprintf("\n");

        for (uint32_t i = 0; i < set->items->num_elements; i++) {
                struct item *probe = set->items->elements[i];
                pprintf("%s", probe->name);

                uint32_t whitespace = max_len - strlen(probe->name);
                for (uint32_t x = 0; x < whitespace; x++)
                        printf(" ");

                for (uint32_t j = 0; j < item->num_events; j++) {
                        double tau = taus->elements[i][j];
                       
                        if (j > 0) {
                                printf("  ");
                                double delta_tau = taus->elements[i][j]
                                        - taus->elements[i][j - 1];
                                delta_tau > 0.0 ? printf("\x1b[32m+%.5f\x1b[0m", delta_tau)
                                        :  printf("\x1b[31m%.5f\x1b[0m", delta_tau);
                                printf("  ");
                        }

                        tau > 0.0 ? printf("\x1b[42m\x1b[30m+%.5f\x1b[0m", tau)
                                : printf("\x1b[41m\x1b[30m%.5f\x1b[0m", tau);


                        if(j == item->num_events - 1) {
                                printf("  ");
                                tau > 0.0 ? printf("\x1b[32m%s\x1b[0m", probe->name)
                                        : printf("\x1b[31m%s\x1b[0m", probe->name);
                        }

                }
                printf("\n");
        }

        dispose_matrix(taus);
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

        if (is_same_vector(a, b))
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
 *************************************************************************/
bool is_same_vector(struct vector *a, struct vector *b)
{
        for (uint32_t i = 0; i < a->size; i++)
                if (a->elements[i] != b->elements[i])
                        return false;

        return true;
}

/**************************************************************************
 * This implements four measures that quantify how much information a word
 * conveys (cf. Frank & Vigliocco, 2011):
 *
 * (1) Syntactic surprisal (Ssyn):
 *
 *     Ssyn(w_i+1) = -log(P(w_i+1|w_1...i))
 *
 * (2) Syntactic entropy reduction (DHsyn):
 *
 *     DHsyn(w_i+1) = Hsyn(i) - Hsyn(i+1)
 *
 *     where 
 *
 *     Hsyn(i) = -sum_(w_1...i,w_i+1...n) P(w_1...i,w_i+1...n|w_1...i)
 *         log(P(w_1...i,w_i+1...n|w_1...i))
 *
 * (3) Semantic surprisal (SSem):
 *
 *     Ssem(w_i+1) = -log((P(sit(w_1...i+1)|w_1...i))
 *
 *     where 
 *
 *     sit(w_1...i) is the disjunction of all situations described by the
 *     first i words (w_1...i) of a sentence
 *
 * (4) Semantic entropy reduction (DHsem):
 *
 *     DHsem(w_i+1) = Hsem(i) - Hsem(i+1)
 *
 *     where
 *
 *     Hsem(i) = -sum_(foreach p_x in S') tau(p_x|sit(w_1...i)) *
 *         log(tau(p_x|sit(w_1...i)))
 *
 *     where S' = {p_x} and mu(p_x) is a situation vector, such that:
 *
 *                 | 0 if x != j
 *     mu_j(p_x) = |
 *                 | 1 if x = j
 *
 *     and where 
 *                             sum_j (mu_j(p_x) * mu_j(sit(w_1...i)))
 *     tau(p_x|sit(w_1...i)) = --------------------------------------
 *                                   sum_j (mu_j(sit(w_1...i)))
 *
 *     such that:
 *
 *     sum(p_x) tau(p_x|sit(w_1...i)) = 1
 *
 *     and hence tau(p_x|sit(w_1...i)) forms a proper probability over p_x.
 *
 * References
 *
 * Frank, S. L. and Vigliocco, G. (2011). Sentence comprehension as mental
 *     simulation: an information-theoretic perspective. Information, 2,
 *     672-696.
 *************************************************************************/
void dss_word_information(struct network *n, struct item *item)
{
        size_t block_size = strlen(item->name) + 1;
        char prefix1[block_size];
        memset(&prefix1, 0, block_size);
        char prefix2[block_size];
        memset(&prefix2, 0, block_size);

        struct vector *sit1 = create_vector(n->output->vector->size);
        struct vector *sit2 = create_vector(n->output->vector->size);

        int32_t *freq_table = frequency_table(n->asp);
        struct set *s = n->asp;

        pprintf("Word    \tSsyn    \tDHsyn   \tSsem    \tDHsem   \n");
        pprintf("========\t========\t========\t========\t========\n");

        /* compute measures for each word in the sentence */
        for (uint32_t i = 0; i < item->num_events; i++) {
                /* reset prefixes */
                strncpy(prefix1, item->name, strlen(item->name));
                strncpy(prefix2, item->name, strlen(item->name));

                /* reset disjunctions of sit(w_1...i) */
                zero_out_vector(sit1);
                zero_out_vector(sit2);

                /* isolate current word, and sentence prefixes */
                uint32_t wp = 0, cp = 0;
                char *word = &prefix2[0];
                for (char *p = item->name; *p != '\0'; p++) {
                        if (i == 0 && cp == 0)
                                prefix1[cp] = '\0';
                        if (*p == ' ') {
                                wp++;
                                if (wp == i) {
                                        prefix1[cp] = '\0';
                                        word = &prefix2[cp + 1];
                                }
                                if (wp == i + 1)
                                        prefix2[cp] = '\0';
                        }
                        cp++;
                }

                /*
                 * Compute for both prefix w_1...i and w_1...i+1:
                 *
                 * 1) The frequency of these prefixes in the active set.
                 *
                 * 2) The disjunction of all situation vectors that are
                 *    consistent with the state of affairs described by
                 *    these prefixes
                 */
                uint32_t freq_prefix1 = 0, freq_prefix2 = 0;
                for (uint32_t j = 0; j < s->items->num_elements; j++) {
                        struct item *ti = s->items->elements[j];
                        struct vector *tv = ti->targets[ti->num_events - 1];
                        /* w_1...i */
                        if (strncmp(ti->name, prefix1, strlen(prefix1)) == 0) {
                                freq_prefix1++;
                                j == 0 ? copy_vector(sit1, tv) : fuzzy_or(sit1, tv);
                        }
                        /* w_1...i+1 */
                        if (strncmp(ti->name, prefix2, strlen(prefix2)) == 0) {
                                freq_prefix2++;
                                j == 0 ? copy_vector(sit2, tv) : fuzzy_or(sit2, tv);
                        }
                }

                /* 
                 * Compute syntactic entropy for prefix w_1...i and
                 * w_1...i+1:
                 *
                 * Hsyn(i) = -sum_(w_1...i,w_i+1...n)
                 *     P(w_1...i,w_i+1...n|w_1...i)
                 *     * log(P(w_1...i,w_i+1...n|w_1...i))
                 */
                double hsyn1 = 0.0, hsyn2 = 0.0;
                for (uint32_t j = 0; j < s->items->num_elements; j++) {
                        /* skip doubles */
                        if (freq_table[j] == -1)
                                continue;
                        struct item *ti = s->items->elements[j];
                        /* w_1...i */
                        if (strncmp(ti->name, prefix1, strlen(prefix1)) == 0)
                                hsyn1 -= ((double)freq_table[j] / freq_prefix1)
                                        * log((double)freq_table[j] / freq_prefix1);
                        /* w_1...i+1 */
                        if (strncmp(ti->name, prefix2, strlen(prefix2)) == 0)
                                hsyn2 -= ((double)freq_table[j] / freq_prefix2)
                                        * log((double)freq_table[j] / freq_prefix2);
                }

                /*
                 * Compute semantic entropy for prefix w_1...i and
                 * w_1...i+1:
                 *
                 * Hsem(i) = -sum_(foreach p_x in S')
                 *     tau(p_x|sit(w_1...i))
                 *     * log(tau(p_x|sit(w_1...i)))
                 *
                 * where
                 *                         sum_j (mu_j(p_x) * mu_j(sit(w_1...i)))
                 * tau(p_x|sit(w_1...i)) = --------------------------------------
                 *                               sum_j (mu_j(sit(w_1...i)))
                 */
                double ssum1 = 0.0, ssum2 = 0.0;
                for (uint32_t j = 0 ; j < sit1->size; j++) {
                        ssum1 += sit1->elements[j];
                        ssum2 += sit2->elements[j];
                }
                double hsem1 = 0.0, hsem2 = 0.0;
                for (uint32_t j = 0 ; j < sit1->size; j++) {
                        double tau1 = sit1->elements[j] / ssum1;
                        double tau2 = sit2->elements[j] / ssum2;
                        if (tau1 > 0.0) hsem1 -= tau1 * log(tau1);
                        if (tau2 > 0.0) hsem2 -= tau2 * log(tau2);
                }
                
                /*
                 * Syntactic surprisal:
                 * 
                 * Ssyn(w_i+1) = -log(P(w_i+1|w_1...i))
                 *     = log(P(w_1...i)) - log(P(w_1...i+1)
                 *     = log(freq(w_1...i)) - log(freq(w_1...i+1))
                 */
                double ssyn = log(freq_prefix1) - log(freq_prefix2);

                /*
                 * Syntactic entropy reduction:
                 *
                 * DHsyn(w_i+1) = Hsyn(i) - Hsyn(i+1)
                 */
                double delta_hsyn = hsyn1 - hsyn2;

                /*
                 * Semantic surprisal:
                 *
                 * Ssem(w_i+1) = -log((P(sit(w_1...i+1)|w_1...i))
                 *     = log(P(sit(w_1...i)) - log(P(sit(w_1...i+1)))
                 *     = log(tau(sit(w_1...i)) - log(tau(sit(w_1...i+1)))
                 */
                double ssem = log(dss_tau_prior(sit1)) - log(dss_tau_prior(sit2));

                /*
                 * Semantic entropy reduction:
                 *
                 * DHsem(w_i+1) = Hsem(i) - Hsem(i+1)
                 */
                double delta_hsem = hsem1 - hsem2;

                /* print measures */
                pprintf("%s\t%f\t%f\t%f\t%f\n", word, ssyn, delta_hsyn, ssem, delta_hsem);
        }

        free(freq_table);

        dispose_vector(sit1);
        dispose_vector(sit2);

        return;
}

/**************************************************************************
 *************************************************************************/
int32_t *frequency_table(struct set *s)
{
        int32_t *freq_table;
        uint32_t n = s->items->num_elements;
        if (!(freq_table = malloc(n * sizeof(int32_t))))
                goto error_out;
        memset(freq_table, 0, n * sizeof(int32_t));

        /* populate frequency table */
        for (uint32_t i = 0; i < n; i++) {
                /* skip seen items */
                if (freq_table[i] == -1)
                        continue;
                struct item *item1 = s->items->elements[i];
                for (uint32_t j = 0; j < n; j++) {
                        struct item *item2 = s->items->elements[j];
                        if (strcmp(item1->name, item2->name) == 0) {
                                freq_table[i]++;
                                if (i != j)
                                        freq_table[j] = -1; /* mark as seen */
                        }
                }
        }

        return freq_table;

error_out:
        return NULL;
}

/**************************************************************************
 *************************************************************************/
void fuzzy_or(struct vector *a, struct vector *b)
{
        for (uint32_t i = 0; i < a->size; i++)
                a->elements[i] = a->elements[i] + b->elements[i]
                        - a->elements[i] * b->elements[i];

        return;
}
