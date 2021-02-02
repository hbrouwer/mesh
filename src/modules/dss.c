/*
 * Copyright 2012-2021 Harm Brouwer <me@hbrouwer.eu>
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

#include "../engine.h"
#include "../error.h"
#include "../main.h"
#include "../math.h"

                /********************************************
                 **** distributed-situation state spaces ****
                 ********************************************/

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
This implements a variety of functions for dealing with Distributed
Situation Space vectors, see:
 
Frank, S. L., Haselager, W. F. G, & van Rooij, I. (2009). Connectionist
        semantic systematicity. Cognition, 110, 358-379.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */ 

void dss_test(struct network *n)
{
        double acs = 0.0;       /* accumulated comprehension score */
        uint32_t ncs = 0;       /* number of comprehension scores */

        struct vector *ov = create_vector(n->output->vector->size);

        cprintf("\n");
        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                struct item *item = n->asp->items->elements[i];
                reset_ticks(n);
                for (uint32_t j = 0; j < item->num_events; j++) {
                        if (j > 0)
                                next_tick(n);
                        clamp_input_vector(n, item->inputs[j]);
                        forward_sweep(n);
                }

                /* comprehension score */
                struct vector *tv = item->targets[item->num_events - 1];
                dss_adjust_output_vector(ov, output_vector(n), tv,
                        n->pars->target_radius, n->pars->zero_error_radius);
                double tau = dss_comprehension_score(tv, ov);
                if (!isnan(tau)) {
                        acs += tau;
                        ncs++;
                }
                tau > 0.0 ? pprintf("%d: \x1b[32m%s: %f\x1b[0m\n",
                                i + 1, item->name, tau)
                          : pprintf("%d: \x1b[31m%s: %f\x1b[0m\n",
                                i + 1, item->name, tau);
        }
        cprintf("\nAverage comprehension score: (%f / %d =) %f\n\n",
                acs, ncs, acs / ncs);

        free_vector(ov);

        return;
}

void dss_scores(struct network *n, struct set *set, struct item *item)
{
        struct matrix *sm = dss_score_matrix(n, set, item);

        cprintf("\n");
        cprintf("Sentence:  \"%s\"\n", item->name);
        cprintf("Semantics: \"%s\"\n", item->meta);
        cprintf("\n");

        uint32_t word_col_len = 20;     /* word column length */
        uint32_t init_col_len = 0;      /* initial column length */

        /* determine initial column length */
        for (uint32_t i = 0; i < set->items->num_elements; i++) {
                struct item *probe = set->items->elements[i];
                uint32_t len = strlen(probe->name);
                if (len > init_col_len)
                        init_col_len = len;
        }
        init_col_len++;

        /* print the words of the sentence as columns */
        size_t block_size = strlen(item->name) + 1;
        char sentence[block_size];
        memset(&sentence, 0, block_size);
        strcpy(sentence, item->name);
        for (uint32_t i = 0; i < init_col_len; i++)
                cprintf(" ");
        char *token = strtok(sentence, " ");
        do {
                cprintf("\x1b[35m%s\x1b[0m", token);
                for (uint32_t i = 0; i < word_col_len - strlen(token); i++)
                        cprintf(" ");
                token = strtok(NULL, " ");
        } while (token);
        cprintf("\n");

        /* print the overall comprehension scores */
        cprintf("\n");
        for (uint32_t i = 0; i < init_col_len; i++)
                cprintf(" ");
        if (!isnan(sm->elements[0][0])) {
                for (uint32_t c = 0; c < sm->cols; c++) {
                        double score = sm->elements[0][c];
                        if (c > 0) {    /* print score delta */
                                double delta = score - sm->elements[0][c - 1];
                                cprintf("  ");
                                delta >= 0.0 ? cprintf("\x1b[32m+%.5f\x1b[0m",
                                                delta)
                                             : cprintf("\x1b[31m%.5f\x1b[0m",
                                                delta);
                                cprintf("  ");
                        }               /* print score */
                        score >= 0.0 ? cprintf("\x1b[42m\x1b[30m+%.5f\x1b[0m",
                                        score)
                                     : cprintf("\x1b[41m\x1b[30m%.5f\x1b[0m",
                                        score);
                }
        } else {
                cprintf("\x1b[41m\x1b[30mcomprehension score undefined: unlawful situation\x1b[0m");
        }
        cprintf("\n");

        /* print scores per probe event */
        cprintf("\n");
        for (uint32_t r = 0; r < set->items->num_elements; r++) {
                struct item *probe = set->items->elements[r];
                cprintf("%s", probe->name);
                uint32_t whitespace = init_col_len - strlen(probe->name);
                for (uint32_t i = 0; i < whitespace; i++)
                        cprintf(" ");
                for (uint32_t c = 0; c < item->num_events; c++) {
                        double score = sm->elements[r + 1][c];
                        if (c > 0) {    /* print score delta */
                                double delta = score - sm->elements[r + 1][c - 1];
                                cprintf("  ");
                                delta >= 0.0 ? cprintf("\x1b[32m+%.5f\x1b[0m",
                                                delta)
                                             : cprintf("\x1b[31m%.5f\x1b[0m",
                                                delta);
                                printf("  ");
                        }               /* print score */
                        score >= 0.0 ? cprintf("\x1b[42m\x1b[30m+%.5f\x1b[0m",
                                        score)
                                     : cprintf("\x1b[41m\x1b[30m%.5f\x1b[0m",
                                        score);
                        if(c == item->num_events - 1) {
                                cprintf("  ");
                                score >= 0.0 ? cprintf("\x1b[32m%s\x1b[0m",
                                                probe->name)
                                             : cprintf("\x1b[31m%s\x1b[0m",
                                                probe->name);
                        }
                }
                cprintf("\n");
        }
        cprintf("\n");

        free_matrix(sm);

        return;
}

void dss_inferences(struct network *n, struct set *set, struct item *item,
        float threshold)
{
        struct matrix *sm = dss_score_matrix(n, set, item);

        cprintf("\n");
        cprintf("Sentence:      \"%s\"\n", item->name);
        cprintf("Semantics:     \"%s\"\n", item->meta);
        cprintf("\n");

        uint32_t c = sm->cols - 1;
        
        /* print overall comprehension score */
        cprintf("Overall score: ");
        double score = sm->elements[0][c];
        if (!isnan(score))
                score >= 0.0 ? cprintf("\x1b[42m\x1b[30m+%.5f\x1b[0m\n", score)
                             : cprintf("\x1b[41m\x1b[30m%.5f\x1b[0m\n", score);
        else
                cprintf("\x1b[41m\x1b[30mcomprehension score undefined: unlawful situation\x1b[0m\n");
        
        cprintf("\n");

        /* print inferences */
        for (uint32_t r = 1; r < sm->rows; r++) {
                struct item *probe = set->items->elements[r - 1];
                score = sm->elements[r][c];
                if (fabs(score) >= fabs(threshold))
                        score >= 0.0 ? cprintf("\x1b[32m[+%.5f]: %s\x1b[0m\n",
                                        probe->name, score)
                                     : cprintf("\x1b[31m[%.5f]: %s\x1b[0m\n",
                                        probe->name, score);
        }
        
        cprintf("\n");

        free_matrix(sm);
}

/*
 * Fills a vector with the output vector adjusted for target radius and zero
 * error radius. The output vector is adjusted in the direction of the
 * target vector (i.e., the inverse of what happens in error computation).
 */
void dss_adjust_output_vector(struct vector *av, struct vector *ov,
        struct vector *tv, double tr, double zr)
{
        for (uint32_t i = 0; i < av->size; i++)
                av->elements[i] = adjust_target(tv->elements[i], ov->elements[i], tr, zr);
}

/*
 * Fills a vector with the comprehension score of each proposition in the
 * specified set, given the output of the model.
 */
void dss_score_vector(struct vector *v, struct network *n, struct set *set)
{
        for (uint32_t i = 0; i < set->items->num_elements; i++) {
                struct item *probe = set->items->elements[i];
                struct vector *pv = probe->targets[0];
                v->elements[i] = dss_comprehension_score(pv, output_vector(n));
        }
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
This construsts a (1+m) x n comprehension score matrix, where m is the
number of events for which a score is computed after processing each of n
words of a sentence. The first row of the matrix contains the scores for the
target event of the current sentence.

           n
    [ . . . . . . ] <-- overall comprehension scores
    [ . . . . . . ] <-- score for event 1
1+m [ . . . . . . ] <-- score for event 2
    [ . . . . . . ] ...
    [ . . . . . . ] <-- score for event n
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

struct matrix *dss_score_matrix(struct network *n, struct set *set,
        struct item *item)
{
        uint32_t rows = set->items->num_elements + 1;
        uint32_t cols = item->num_events;
        struct matrix *sm = create_matrix(rows, cols);
        struct vector *ov = create_vector(n->output->vector->size);

        reset_ticks(n);
        for (uint32_t i = 0; i < item->num_events; i++) {
                if (i > 0)
                        next_tick(n);
                clamp_input_vector(n, item->inputs[i]);
                forward_sweep(n);

                /*
                 * Compute overall comprehension score, as well as
                 * comprehension scores per probe event.
                 */
                struct vector *tv = item->targets[item->num_events - 1];
                dss_adjust_output_vector(ov, output_vector(n), tv,
                        n->pars->target_radius, n->pars->zero_error_radius);                
                sm->elements[0][i] = dss_comprehension_score(tv, ov);
                for (uint32_t j = 0; j < set->items->num_elements; j++) {
                        struct item *probe = set->items->elements[j];
                        struct vector *pv = probe->targets[0];
                        sm->elements[j + 1][i] = dss_comprehension_score(pv, ov);
                }
        }

        free_vector(ov);

        return sm;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
This computes the comprehension score (Frank et al., 2009), which is defined
as:

                        | tau(a|z) - tau(a)
                        | ----------------- , if tau(a|z) > tau(a)
                        |    1 - tau(a)
        comprehension = |
                        | tau(a|z) - tau(a)
                        | ----------------- , otherwise
                        |      tau(a)

where tau(a|z) is the conditional belief of a given z, and tau(a) is the
prior belief in a.

If tau(a|z) = 1, the comprehension score is maximal: +1. On the other hand,
if tau(a|z) = 0, the comprehension score is minimal: -1. Intuitively, a
positive comprehension score is a measure of how much uncertainty in event a
is taken away by z, whereas a negative comprehension score measures how much
certainty in event a is taken away by z.

References

Frank, S. L., Haselager, W. F. G, & van Rooij, I. (2009). Connectionist
        semantic systematicity. Cognition, 110, 358-379.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

double dss_comprehension_score(struct vector *a, struct vector *z)
{
        double tau_a_given_z = dss_tau_conditional(a, z); /* tau(a|z) */
        double tau_a         = dss_tau_prior(a);          /* tau(a)   */

        /* unlawful event */
        if (tau_a == 0.0)
                return NAN;

        double cs = 0.0;
        if (tau_a_given_z > tau_a)
                cs = (tau_a_given_z - tau_a) / (1.0 - tau_a);
        else
                cs = (tau_a_given_z - tau_a) / tau_a;

        return cs;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Prior belief in a:

        tau(a) = 1/n sum_i u_i(a)
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

double dss_tau_prior(struct vector *a)
{
        double tau = 0.0;
        for (uint32_t i = 0; i < a->size; i++)
                tau += dss_clip_unit(a->elements[i]);

        return tau / a->size;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Conjunction belief in a and b:

        tau(a^b) = 1/n sum_i u_i(a) * u_i(b)
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

double dss_tau_conjunction(struct vector *a, struct vector *b)
{
        double tau = 0.0;
        if (is_same_vector(a, b))
                tau = dss_tau_prior(a);
        else {
                for (uint32_t i = 0; i < a->size; i++)
                        tau += dss_clip_unit(a->elements[i]) 
                                * dss_clip_unit(b->elements[i]);
                tau /= a->size;
        }
        
        return tau;
}

double dss_clip_unit(double u)
{
        return maximum(minimum(u, 1.0), 0.0);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Conditional belief in a given b:
 
        tau(a|b) = tau(a^b) / tau(b)
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

double dss_tau_conditional(struct vector *a, struct vector *b)
{
        return dss_tau_conjunction(a, b) / dss_tau_prior(b);
}

bool is_same_vector(struct vector *a, struct vector *b)
{
        for (uint32_t i = 0; i < a->size; i++)
                if (a->elements[i] != b->elements[i])
                        return false;

        return true;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
This implements four offline measures from Frank & Vigliocco (2011) that
quantify how much information a word conveys:

(1) Syntactic surprisal (Ssyn):

        Ssyn(w_i+1) = -log(P(w_i+1|w_1...i))

(2) Syntactic entropy reduction (DHsyn):

        DHsyn(w_i+1) = Hsyn(i) - Hsyn(i+1)

        where 

        Hsyn(i) = -sum_(w_1...i,w_i+1...n) P(w_1...i,w_i+1...n|w_1...i)
                * log(P(w_1...i,w_i+1...n|w_1...i))
 
(3) Semantic surprisal (SSem):

        Ssem(w_i+1) = -log((P(sit(w_1...i+1)|w_1...i))

        where 

        sit(w_1...i) is the disjunction of all situations described by the
        first i words (w_1...i) of a sentence
 
(4) Semantic entropy reduction (DHsem):

        DHsem(w_i+1) = Hsem(i) - Hsem(i+1)

        where
 
        Hsem(i) = -sum_(foreach p_x in S') tau(p_x|sit(w_1...i))
                * log(tau(p_x|sit(w_1...i)))
 
        where S' = {p_x} and mu(p_x) is a situation vector, such that:

                    | 0 if x != j
        mu_j(p_x) = |
                    | 1 if x = j
                     
        and where 
                                sum_j (mu_j(p_x) * mu_j(sit(w_1...i)))
        tau(p_x|sit(w_1...i)) = --------------------------------------
                                      sum_j (mu_j(sit(w_1...i)))
 
        such that:

        sum(p_x) tau(p_x|sit(w_1...i)) = 1

        and hence tau(p_x|sit(w_1...i)) forms a proper probability over p_x.

In addition, two additional metrics are computed:

(5) Online surprisal: This is the same as (3), but sit(w_1...i+1) and
sit(w_1...i) are the DSS vectors at the output layer of the network at after
processing w1...i+1 (DSS_i+1) and w1...i (DSS_i), respectively.

(6) Online entropy reduction: This is the same as (4), but sit(w_1...i) is
the DSS vector at the output layer of the network after processing w1...i
(DSS_i).

These metrics are returned in an m x 6 matrix. The m rows of this matrix
represent the words of the current sentence, and the 7 columns contain
respectively the Ssyn, DHsyn, SSem, DHsem, Sonl, and DHonl value for each of
these words.

References

Frank, S. L. and Vigliocco, G. (2011). Sentence comprehension as mental
        simulation: an information-theoretic perspective. Information, 2,
        672-696.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

struct matrix *dss_word_info_matrix(struct network *n,
        struct set *s, struct item *item, int32_t *freq_table)
{
        struct matrix *im = create_matrix(item->num_events, 6);

                /**************************
                 **** offline measures ****
                 **************************/

        size_t block_size = strlen(item->name) + 1;
        char prefix1[block_size];
        memset(&prefix1, 0, block_size);
        char prefix2[block_size];
        memset(&prefix2, 0, block_size);

        struct vector *sit1 = create_vector(n->output->vector->size);
        struct vector *sit2 = create_vector(n->output->vector->size);

        /* compute measures for each word in the sentence */
        for (uint32_t i = 0; i < item->num_events; i++) {
                /* reset prefixes */
                strcpy(prefix1, item->name);
                strcpy(prefix2, item->name);

                /* reset disjunctions of sit(w_1...i) */
                zero_out_vector(sit1);
                zero_out_vector(sit2);

                /* isolate current word, and sentence prefixes */
                uint32_t wp = 0, cp = 0;
                for (char *p = item->name; *p != '\0'; p++) {
                        if (i == 0 && cp == 0)
                                prefix1[cp] = '\0';
                        if (*p == ' ') {
                                wp++;
                                if (wp == i)
                                        prefix1[cp] = '\0';
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
                                j == 0 ? copy_vector(tv, sit1) 
                                       : fuzzy_or(sit1, tv);
                        }
                        /* w_1...i+1 */
                        if (strncmp(ti->name, prefix2, strlen(prefix2)) == 0) {
                                freq_prefix2++;
                                j == 0 ? copy_vector(tv, sit2) 
                                       : fuzzy_or(sit2, tv);
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
                                hsyn1 -= ((double)freq_table[j]
                                                / freq_prefix1)
                                        * log((double)freq_table[j] 
                                                / freq_prefix1);
                        /* w_1...i+1 */
                        if (strncmp(ti->name, prefix2, strlen(prefix2)) == 0)
                                hsyn2 -= ((double)freq_table[j]
                                                / freq_prefix2)
                                        * log((double)freq_table[j]
                                                / freq_prefix2);
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
                 *                              sum_j (mu_j(p_x) 
                 *                           * mu_j(sit(w_1...i)))
                 * tau(p_x|sit(w_1...i)) = --------------------------
                 *                         sum_j (mu_j(sit(w_1...i)))
                 *
                 * Note: As this defines a probability distribution over
                 *     the observations that constitute the DSS by iterating
                 *     over the individual dimensions, there should be no
                 *     duplicate observations. Duplicates would require
                 *     identification of unique observations in order to
                 *     obtain a proper probability distribution.
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
                 *
                 * Note: This assumes that sit(w_1...i+1) |= sit(w_1...i),
                 *     and hence that: tau(sit(w_1...i+1))
                 *     = tau(sit(w_1...i+1) & sit(w_1...i))
                 */
                double ssem = log(dss_tau_prior(sit1))
                        - log(dss_tau_prior(sit2));

                /*
                 * Semantic entropy reduction:
                 *
                 * DHsem(w_i+1) = Hsem(i) - Hsem(i+1)
                 */
                double delta_hsem = hsem1 - hsem2;

                /* add scores to matrix */
                im->elements[i][0] = ssyn;
                im->elements[i][1] = delta_hsyn;
                im->elements[i][2] = ssem;
                im->elements[i][3] = delta_hsem;
        }

        free_vector(sit1);
        free_vector(sit2);

                /*************************
                 **** online measures ****
                 *************************/
        
        /* 
         * Output vector and previous output vector. At time-step t=0, we
         * bootstrap this by using the unit vector.
         */
        struct vector *ov = create_vector(n->output->vector->size);
        struct vector *pv = create_vector(n->output->vector->size);
        fill_vector_with_value(pv, 1.0);
        fill_vector_with_value(pv, 1.0 / euclidean_norm(pv));

        reset_ticks(n);
        for (uint32_t i = 0; i < item->num_events; i++) {
                if (i > 0)
                        next_tick(n);
                clamp_input_vector(n, item->inputs[i]);
                forward_sweep(n);
                struct vector *tv = item->targets[item->num_events - 1];
                dss_adjust_output_vector(ov, output_vector(n), tv,
                        n->pars->target_radius, n->pars->zero_error_radius);
                for (uint32_t i = 0; i < ov->size; i++)
                        ov->elements[i] = dss_clip_unit(ov->elements[i]);

                /*
                 * Compute semantic entropy for prefix w_1...i and
                 * w_1...i+1:
                 *
                 * Hsem(i) = -sum_(foreach p_x in S')
                 *     tau(p_x|DSS_i)
                 *     * log(tau(p_x|DSS_i))
                 *
                 * where
                 *                  sum_j (mu_j(p_x) * mu_j(DSS_i))
                 * tau(p_x|DSS_i) = -------------------------------
                 *                         sum_j (mu_j(DSS_i))
                 * 
                 * where DSS_i is the output vector of the network at
                 * after processing w_1...i.
                 *
                 * Note: As this defines a probability distribution over
                 *     the observations that constitute the DSS by iterating
                 *     over the individual dimensions, there should be no
                 *     duplicate observations. Duplicates would require
                 *     identification of unique observations in order to
                 *     obtain a proper probability distribution.
                 */
                double ssum1 = 0.0, ssum2 = 0.0;
                for (uint32_t j = 0 ; j < ov->size; j++) {
                        ssum1 += pv->elements[j];
                        ssum2 += ov->elements[j];
                }
                double hsem1 = 0.0, hsem2 = 0.0;
                for (uint32_t j = 0 ; j < ov->size; j++) {
                        double tau1 = pv->elements[j] / ssum1;
                        double tau2 = ov->elements[j] / ssum2;
                        if (tau1 > 0.0) hsem1 -= tau1 * log(tau1);
                        if (tau2 > 0.0) hsem2 -= tau2 * log(tau2);
                }

                /*
                 * Online surprisal
                 * 
                 * Sonl = -log(tau(DSS_i+1)|DSS_i)
                 */
                double sonl = -log(dss_tau_conditional(ov, pv));

                /*
                 * Online entropy reduction:
                 *
                 * DHonl(w_i+1) = Honl(i) - Honl(i+1)
                 */
                double delta_honl = hsem1 - hsem2;

                /* add scores to matrix */
                im->elements[i][4] = sonl;
                im->elements[i][5] = delta_honl;

                copy_vector(ov, pv);
        }

        free_vector(pv);
        free_vector(ov);

        return im;
}

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

void fuzzy_or(struct vector *a, struct vector *b)
{
        for (uint32_t i = 0; i < a->size; i++)
                a->elements[i] = a->elements[i] + b->elements[i]
                        - a->elements[i] * b->elements[i];

        return;
}

void dss_word_info(struct network *n, struct set *s,
        struct item *item)
{
        int32_t *freq_table = frequency_table(s);
        struct matrix *im = dss_word_info_matrix(n, s,
                item, freq_table);
        
        size_t block_size = strlen(item->name) + 1;
        char sentence[block_size];
        memset(&sentence, 0, block_size);
        strcpy(sentence, item->name);

        uint32_t col_len = 20;

        /* print the words of the sentence */
        cprintf("\n");
        for (uint32_t i = 0; i < col_len; i++)
                cprintf(" ");
        char *token = strtok(sentence, " ");
        do {
                cprintf("\x1b[35m%s\x1b[0m", token);
                for (uint32_t i = 0; i < col_len - strlen(token); i++)
                        cprintf(" ");
                token = strtok(NULL, " ");
        } while (token);
        cprintf("\n");

        /* print word information metrics */
        cprintf("\n");
        for (uint32_t c = 0; c < im->cols; c++) {
                switch (c) {
                case 0: /* syntactic surprisal */
                        cprintf("Ssyn ");
                        break;
                case 1: /* syntactic entropy reduction */
                        cprintf("DHsyn");
                        break;
                case 2: /* semantic surprisal */
                        cprintf("Ssem ");
                        break;
                case 3: /* semantic entropy reduction */
                        cprintf("DHsem");
                        break;
                case 4: /* online surprisal */
                        cprintf("\n");
                        cprintf("Sonl ");
                        break;
                case 5: /* online entropy reduction */
                        cprintf("DHonl");
                        break;
                }
                for (uint32_t i = 0; i < col_len - 5; i++)
                        cprintf(" ");
                for (uint32_t r = 0; r < item->num_events; r++) {
                        cprintf("%.5f", im->elements[r][c]);
                        for (uint32_t i = 0; i < col_len - 7; i++)
                                cprintf(" ");      
                }
                cprintf("\n");
        }
        cprintf("\n");

        free_matrix(im);
        free(freq_table);

        return;
}

void dss_write_word_info(struct network *n, struct set *s,
        char *filename)
{
        FILE *fd;
        if (!(fd = fopen(filename, "w")))
                goto error_out;
        int32_t *freq_table = frequency_table(s);
        
        cprintf("\n");
        fprintf(fd, "\"ItemId\",\"ItemName\",\"ItemMeta\",\"WordPos\",\"Ssyn\",\"DHsyn\",\"Ssem\",\"DHsem\",\"Sonl\",\"DHonl\"\n");
        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                struct item *item =
                        n->asp->items->elements[i];
                struct matrix *im =
                        dss_word_info_matrix(n, s, item, freq_table);
                for (uint32_t j = 0; j < item->num_events; j++) {
                        fprintf(fd, "%d,\"%s\",\"%s\",%d",
                                i + 1, item->name, item->meta, j + 1);
                        for (uint32_t x = 0; x < im->cols; x++)
                                fprintf(fd, ",%f", im->elements[j][x]);
                        fprintf(fd, "\n");
                }
                pprintf("%d: %s\n", i + 1, item->name);
                free_matrix(im);
        }
        cprintf("\n");

        free(freq_table);
        fclose(fd);

        return;

error_out:
        perror("[dss_write_word_info()]");
        return;
}

void reset_dcs_vectors(struct network *n)
{
        for (uint32_t i = 0; i < n->groups->num_elements; i++) {
                struct group *g = n->groups->elements[i];
                if (g->pars->dcs_set) {
                        struct group *ng = find_network_group_by_name(n, g->name);
                        zero_out_vector(ng->vector);
                }
        }

}

void update_dcs_vectors(struct network *n)
{
        for (uint32_t i = 0; i < n->groups->num_elements; i++) {
                struct group *g = n->groups->elements[i];
                if (g->pars->dcs_set) {
                        struct group *ng = find_network_group_by_name(n, g->name);
                        dss_score_vector(ng->vector, n, ng->pars->dcs_set);
                        /*
                        for (uint32_t j = 0; j < ng->vector->size; j++)
                                if (ng->vector->elements[j] < 0.0)
                                        ng->vector->elements[j] = 0.0;
                                        */
                }
        }
}
