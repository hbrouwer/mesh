/*
 * Copyright 2012-2018 Harm Brouwer <me@hbrouwer.eu>
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
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>

#include "act.h"
#include "main.h"
#include "similarity.h"

static bool keep_running = true;

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Construct a similarity matrix. The rows of this matrix represent the output
vectors for each item the active set, the columns the target vector for each
item, and the cells the similarity between each output and target vector:

                           target:                        
                   |   A   |   B   |   C
                ----------------------------
                A  |  .99  |  .26  |  .30  |
                ----------------------------
        output: B  |  .31  |  .97  |  .12  |
                ----------------------------
                C  |  .44  |  .15  |  .98  |
                ----------------------------
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

struct matrix *similarity_matrix(struct network *n)
{
        struct sigaction sa;
        sa.sa_handler = sm_signal_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;
        sigaction(SIGINT, &sa, NULL);

        keep_running = true;

        struct matrix *sm = NULL;
        switch (n->type) {
        case ntype_ffn: /* fall through */
        case ntype_srn:
                sm = ffn_network_sm(n);
                break;
        case ntype_rnn:
                sm = rnn_network_sm(n);
                break;
        }

        sa.sa_handler = SIG_DFL;
        sigaction(SIGINT, &sa, NULL);

        return sm;
}

struct matrix *ffn_network_sm(struct network *n)
{
        uint32_t d = n->asp->items->num_elements;
        struct matrix *sm = create_matrix(d, d);

        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                if (!keep_running)
                        goto out;
                 struct item *item = n->asp->items->elements[i];

                if (n->type == ntype_srn)
                        reset_context_groups(n);
                for (uint32_t j = 0; j < item->num_events; j++) {
                        if (j > 0 && n->type == ntype_srn)
                                shift_context_groups(n);
                        copy_vector(n->input->vector, item->inputs[j]);
                        feed_forward(n, n->input);
                        
                        /* only compute distance metrics for last event */
                        if (!(item->targets[j] && j == item->num_events - 1))
                                continue;
                        similarity_scores(n, n->output, i, sm);
                }
        }

out:
        return sm;
}

struct matrix *rnn_network_sm(struct network *n)
{
        struct rnn_unfolded_network *un = n->unfolded_net;

        uint32_t d = n->asp->items->num_elements;
        struct matrix *sm = create_matrix(d, d);

        /* test network on all items in the current set */
        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                if (!keep_running)
                        goto out;
                struct item *item = n->asp->items->elements[i];

                reset_recurrent_groups(un->stack[un->sp]);
                for (uint32_t j = 0; j < item->num_events; j++) {
                        copy_vector(
                                un->stack[un->sp]->input->vector,
                                item->inputs[j]);
                        feed_forward(
                                un->stack[un->sp],
                                un->stack[un->sp]->input);

                        /* only compute distance metrics for last event */
                        if (!(item->targets[j] && j == item->num_events - 1))
                                goto next_tick;
                        similarity_scores(n, un->stack[un->sp]->output, i, sm);

next_tick:
                        shift_pointer_or_stack(n);
                }
        }

out:
        return sm;
}

void similarity_scores(struct network *n, struct group *output,
        uint32_t item_num, struct matrix *sm)
{
        for (uint32_t x = 0; x < n->asp->items->num_elements; x++) {
                struct item *ci = n->asp->items->elements[x];
                struct vector *tv = ci->targets[ci->num_events - 1];
                if (!tv)
                        continue;
                sm->elements[item_num][x] = n->similarity_metric(
                        output->vector, tv);
        }
}

void print_sm_summary(struct network *n, bool print_sm, bool pprint,
        enum color_scheme scheme)
{
        struct matrix *sm = similarity_matrix(n);
        
        if (print_sm) {
                cprintf("\nOutput-target similarity matrix:\n\n");
                pprint ? pprint_matrix(sm, scheme) : print_matrix(sm);
        }

        /*
         * Compute mean similarity, and its standard deviation. Also,
         * determine how many items reached threshold.
         */
        uint32_t tr = n->asp->items->num_elements;
        double sim_mean = 0.0, sim_sd = 0.0;
        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                double s = sm->elements[i][i];
                for (uint32_t x = 0; x < n->asp->items->num_elements; x++) {
                        if (!keep_running)
                                return;
                        if (sm->elements[i][x] > s) {
                                tr--; /* note: we count down */
                                break;
                        }
                }
                sim_mean += s;
        }
        sim_mean /= n->asp->items->num_elements;
        for (uint32_t i = 0; i < sm->rows; i++)
                sim_sd += pow(sm->elements[i][i] - sim_mean, 2.0);
        sim_sd = sqrt(sim_sd / n->asp->items->num_elements);

        cprintf("\n");
        cprintf("Similarity statistics:\n");
        cprintf("\n");
        cprintf("Number of items: \t\t %d\n",
                        n->asp->items->num_elements);
        cprintf("Mean similarity: \t\t %lf\n",
                        sim_mean);
        cprintf("SD of similarity:\t\t %lf\n",
                        sim_sd);
        cprintf("# Items reached threshold: \t %d (%.2lf%%)\n",
                        tr, ((double)tr / n->asp->items->num_elements) * 100.0);
        cprintf("\n");
        
        free_matrix(sm);
}

void sm_signal_handler(int32_t signal)
{
        cprintf("Similarity matrix computation interrupted. Abort [y/n]");
        int32_t c = getc(stdin);
        getc(stdin); /* get newline */
        if (c == 'y' || c == 'Y')
                keep_running = false;
}
