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
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>

#include "act.h"
#include "main.h"
#include "pprint.h"
#include "similarity.h"

static bool keep_running = true;

void similarity_matrix(struct network *n, bool print, bool pprint,
        uint32_t scheme)
{
        struct sigaction sa;
        sa.sa_handler = sm_signal_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;
        sigaction(SIGINT, &sa, NULL);

        keep_running = true;

        if (n->type == NTYPE_FFN)
                ffn_network_sm(n, print, pprint, scheme);
        if (n->type == NTYPE_SRN)
                ffn_network_sm(n, print, pprint, scheme);
        if (n->type == NTYPE_RNN)
                rnn_network_sm(n, print, pprint, scheme);

        sa.sa_handler = SIG_DFL;
        sigaction(SIGINT, &sa, NULL);
}

void ffn_network_sm(struct network *n, bool print, bool pprint,
        uint32_t scheme)
{
        uint32_t d = n->asp->items->num_elements;
        struct matrix *sm = create_matrix(d, d);

        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                struct item *item = n->asp->items->elements[i];

                /* abort after signal */
                if (!keep_running)
                        return;

                if (n->type == NTYPE_SRN)
                        reset_context_groups(n);
                for (uint32_t j = 0; j < item->num_events; j++) {
                        /* feed activation forward */
                        if (j > 0 && n->type == NTYPE_SRN)
                                shift_context_groups(n);
                        copy_vector(n->input->vector, item->inputs[j]);
                        feed_forward(n, n->input);
                        
                        /* only compute distance metrics for last event */
                        if (!(item->targets[j] && j == item->num_events - 1))
                                continue;

                        /* compute distance metric */
                        struct group *g = n->output;
                        for (uint32_t x = 0; x < n->asp->items->num_elements; x++) {
                                struct item *citem = n->asp->items->elements[x];
                                struct vector *tv = citem->targets[citem->num_events - 1];
                                if (!tv)
                                        continue;
                                sm->elements[i][x] = n->similarity_metric(g->vector, tv);
                        }
                }
        }

        print_sm_summary(n, sm, print, pprint, scheme);

        dispose_matrix(sm);
}

void rnn_network_sm(struct network *n, bool print, bool pprint,
        uint32_t scheme)
{
        struct rnn_unfolded_network *un = n->unfolded_net;

        uint32_t d = n->asp->items->num_elements;
        struct matrix *sm = create_matrix(d, d);

        /* test network on all items in the current set */
        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                struct item *item = n->asp->items->elements[i];

                /* abort after signal */
                if (!keep_running)
                        return;

                reset_recurrent_groups(un->stack[un->sp]);
                for (uint32_t j = 0; j < item->num_events; j++) {
                        /* feed activation forward */
                        copy_vector(un->stack[un->sp]->input->vector, item->inputs[j]);
                        feed_forward(un->stack[un->sp], un->stack[un->sp]->input);

                        /* only compute distance metrics for last event */
                        if (!(item->targets[j] && j == item->num_events - 1))
                                goto shift_stack;

                        /* compute distance metric */
                        struct group *g = un->stack[un->sp]->output;
                        for (uint32_t x = 0; x < n->asp->items->num_elements; x++) {
                                struct item *citem = n->asp->items->elements[x];
                                struct vector *tv = citem->targets[citem->num_events - 1];
                                if (!tv)
                                        continue;
                                sm->elements[i][x] = n->similarity_metric(g->vector, tv);
                        }

shift_stack:
                        un->sp == un->stack_size - 1
                                ? rnn_shift_stack(un)
                                : un->sp++;
                }
        }

        print_sm_summary(n, sm, print, pprint, scheme);
        
        dispose_matrix(sm);
}

void print_sm_summary(struct network *n, struct matrix *sm, bool print,
        bool pprint, uint32_t scheme)
{
        if (print) {
                cprintf("Output-target similarity matrix:\n\n");
                pprint == true
                        ? pprint_matrix(sm, scheme)
                        : print_matrix(sm);
                cprintf("\n");
        }

        cprintf("\n");
        cprintf("Similarity statistics:\n");

        /*
         * Compute mean similarity, and its standard deviation. Also,
         * determine how many items reached threshold.
         */
        uint32_t tr = n->asp->items->num_elements;
        double sim_mean = 0.0, sim_sd = 0.0;
        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                double s = sm->elements[i][i];
                for (uint32_t x = 0; x < n->asp->items->num_elements; x++) {
                        /* abort after signal */
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
        cprintf("Number of items: \t\t %d\n",
                        n->asp->items->num_elements);
        cprintf("Mean similarity: \t\t %lf\n",
                        sim_mean);
        cprintf("SD of similarity:\t\t %lf\n",
                        sim_sd);
        cprintf("# Items reached threshold: \t %d (%.2lf%%)\n",
                        tr, ((double)tr / n->asp->items->num_elements) * 100.0);
        cprintf("\n");
}

void sm_signal_handler(int32_t signal)
{
        cprintf("Similarity matrix computation interrupted. Abort [y/n]");
        int32_t c = getc(stdin);
        getc(stdin); /* get newline */
        if (c == 'y' || c == 'Y')
                keep_running = false;
}
