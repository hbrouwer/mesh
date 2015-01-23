/*
 * classify.c
 *
 * Copyright 2012-2015 Harm Brouwer <me@hbrouwer.eu>
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
#include "classify.h"
#include "main.h"
#include "pprint.h"

static bool keep_running = true;

/**************************************************************************
 *************************************************************************/
void confusion_matrix(struct network *n, bool print, bool pprint,
                uint32_t scheme)
{
        struct sigaction sa;
        sa.sa_handler = cm_signal_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;
        sigaction(SIGINT, &sa, NULL);

        keep_running = true;

        if (n->type == TYPE_FFN)
                ffn_network_cm(n, print, pprint, scheme);
        if (n->type == TYPE_SRN)
                ffn_network_cm(n, print, pprint, scheme);
        if (n->type == TYPE_RNN)
                rnn_network_cm(n, print, pprint, scheme);

        sa.sa_handler = SIG_DFL;
        sigaction(SIGINT, &sa, NULL);
}

/**************************************************************************
 *************************************************************************/
void ffn_network_cm(struct network *n, bool print, bool pprint,
                uint32_t scheme)
{
        uint32_t d = n->output->vector->size;
        struct matrix *cm = create_matrix(d, d);

        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                struct item *item = n->asp->items->elements[i];

                /* abort after signal */
                if (!keep_running)
                        return;

                if (n->type == TYPE_SRN)
                        reset_context_groups(n);
                for (uint32_t j = 0; j < item->num_events; j++) {
                        /* feed activation forward */
                        if (j > 0 && n->type == TYPE_SRN)
                                shift_context_groups(n);
                        copy_vector(n->input->vector, item->inputs[j]);
                        feed_forward(n, n->input);
                        
                        /* only classify last event */
                        if (!(item->targets[j] && j == item->num_events - 1))
                                continue;

                        /* classify */
                        struct vector *ov = n->output->vector;
                        struct vector *tv = item->targets[j];

                        uint32_t t = 0, o = 0;
                        for (uint32_t x = 0; x < ov->size; x++) {
                                if (tv->elements[x] > tv->elements[t]) t = x;
                                if (ov->elements[x] > ov->elements[o]) o = x;
                        }
                        cm->elements[t][o]++;

                }
        }

        print_cm_summary(n, cm, print, pprint, scheme);

        dispose_matrix(cm);
}

/**************************************************************************
 *************************************************************************/
void rnn_network_cm(struct network *n, bool print, bool pprint,
                uint32_t scheme)
{
        struct rnn_unfolded_network *un = n->unfolded_net;

        uint32_t d = n->output->vector->size;
        struct matrix *cm = create_matrix(d, d);

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

                        /* only classify metrics for last event */
                        if (!(item->targets[j] && j == item->num_events - 1))
                                goto shift_stack;

                        /* classify */
                        struct vector *ov = n->output->vector;
                        struct vector *tv = item->targets[j];

                        uint32_t t = 0, o = 0;
                        for (uint32_t x = 0; x < ov->size; x++) {
                                if (tv->elements[x] > tv->elements[t]) t = x;
                                if (ov->elements[x] > ov->elements[o]) o = x;
                        }
                        cm->elements[t][o]++;

shift_stack:
                        un->sp == un->stack_size - 1 ? rnn_shift_stack(un)
                                : un->sp++;
                }
        }

        print_cm_summary(n, cm, print, pprint, scheme);

        dispose_matrix(cm);
}

/**************************************************************************
 *************************************************************************/
void print_cm_summary(struct network *n, struct matrix *cm, bool print,
                bool pprint, uint32_t scheme)
{
        if (print) {
                pprintf("Confusion matrix (actual x predicted):\n\n");
                pprint == true ? pprint_matrix(cm, scheme)
                        : print_matrix(cm);
        }
        pprintf("Classification statistics:\n");
        pprintf("\n");

        /* row and column totals */
        struct vector *rows = create_vector(cm->rows);
        struct vector *cols = create_vector(cm->cols);

        for (uint32_t r = 0; r < cm->rows; r++) {
                for (uint32_t c = 0; c < cm->cols; c++) {
                        rows->elements[r] += cm->elements[r][c];
                        cols->elements[c] += cm->elements[r][c];
                }
        }

        /* compute statistics */
        double cc = 0.0, ic = 0.0, pr = 0.0, rc = 0.0;
        for (uint32_t r = 0; r < cm->rows; r++) {
                for (uint32_t c = 0; c < cm->cols; c++) {
                        if (r == c) {
                                cc += cm->elements[r][c];
                                if (cols->elements[c] > 0)
                                        pr += cm->elements[r][c] / cols->elements[c];
                                if (rows->elements[r] > 0)
                                        rc += cm->elements[r][c] / rows->elements[r];
                        } else {
                                ic += cm->elements[r][c];
                        }
                }
        }

        pr /= cols->size;
        rc /= rows->size;

        double beta = 1.0; // TODO: make beta a parameter
        double fs = (1.0 + pow(beta,2.0))  * (pr * rc) / ((pr * pow(beta,2.0)) + rc);

        /* report statistics */
        pprintf("Accurracy:\t\t%f\n", cc / (cc + ic));
        pprintf("Error rate:\t%f\n", ic / (cc + ic));
        pprintf("Precision:\t\t%f\n", pr);
        pprintf("Recall:\t\t%f\n", rc);
        pprintf("F(%.2f)-score:\t%f\n", beta, fs);
        
        dispose_vector(rows);
        dispose_vector(cols);
}

/**************************************************************************
 *************************************************************************/
void cm_signal_handler(int32_t signal)
{
        cprintf("Confusion matrix computation interrupted. Abort [y/n]");
        int32_t c = getc(stdin);
        getc(stdin); /* get newline */
        if (c == 'y' || c == 'Y')
                keep_running = false;
}
