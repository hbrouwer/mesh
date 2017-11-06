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

#include <signal.h>
#include <stdbool.h>
#include <stdio.h>

#include "act.h"
#include "main.h"
#include "pprint.h"
#include "test.h"

static bool keep_running = true;

void test_network(struct network *n)
{
        struct sigaction sa;
        sa.sa_handler = testing_signal_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;
        sigaction(SIGINT, &sa, NULL);

        keep_running = true;

        if (n->type == NTYPE_FFN)
                test_ffn_network(n);
        if (n->type == NTYPE_SRN)
                test_ffn_network(n);
        if (n->type == NTYPE_RNN)
                test_rnn_network(n);

        sa.sa_handler = SIG_DFL;
        sigaction(SIGINT, &sa, NULL);
}

void test_ffn_network(struct network *n)
{
        n->status->error = 0.0;
        uint32_t threshold_reached = 0;

        /* test network on all items in the current set */
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

                        /* only compute network error for last event */
                        if (!(item->targets[j] && j == item->num_events - 1))
                                continue;

                        struct group *g = n->output;
                        struct vector *tv = item->targets[j];
                        double tr = n->target_radius;
                        double zr = n->zero_error_radius;

                        /* compute error */
                        double error = n->output->err_fun->fun(g, tv, tr, zr);
                        n->status->error += error;
                        if (error <= n->error_threshold)
                                threshold_reached++;
                }
        }

        print_testing_summary(n, threshold_reached);
}

void test_rnn_network(struct network *n)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        n->status->error = 0.0;
        uint32_t threshold_reached = 0;

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

                        /*
                         * Only compute error if there is a target for the
                         * current event.
                         */
                        if (!item->targets[j])
                                goto shift_stack;

                        struct group *g = un->stack[un->sp]->output;
                        struct vector *tv = item->targets[j];
                        double tr = n->target_radius;
                        double zr = n->zero_error_radius;

                        /* compute error */
                        double error = n->output->err_fun->fun(g, tv, tr, zr);
                        n->status->error += error;
                        if (error < n->error_threshold)
                                threshold_reached++;

shift_stack:
                        un->sp == un->stack_size - 1
                                ? rnn_shift_stack(un)
                                : un->sp++;
                }
        }

        print_testing_summary(n, threshold_reached);
}

void test_network_with_item(struct network *n, struct item *item,
        bool pprint, uint32_t scheme)
{
        if (n->type == NTYPE_FFN)
                test_ffn_network_with_item(n, item, pprint, scheme);
        if (n->type == NTYPE_SRN)
                test_ffn_network_with_item(n, item, pprint, scheme);
        if (n->type == NTYPE_RNN)
                test_rnn_network_with_item(n, item, pprint, scheme);
}

void test_ffn_network_with_item(struct network *n, struct item *item,
        bool pprint, uint32_t scheme)
{
        n->status->error = 0.0;

        cprintf("\n");
        cprintf("Name:   \"%s\"\n", item->name);
        cprintf("Meta:   \"%s\"\n", item->meta);
        cprintf("Events: %d\n", item->num_events);
        cprintf("\n");
        cprintf("(E: Event; I: Input; T: Target; O: Output)\n");

        if (n->type == NTYPE_SRN)
                reset_context_groups(n);
        for (uint32_t i = 0; i < item->num_events; i++) {
                /* print event number, and input vector */
                cprintf("\n");
                cprintf("E: %d\n", i + 1);
                cprintf("I: ");
                pprint == true ? pprint_vector(item->inputs[i], scheme)
                        : print_vector(item->inputs[i]);

                /* feed activation forward */
                if (i > 0 && n->type == NTYPE_SRN)
                        shift_context_groups(n);
                copy_vector(n->input->vector, item->inputs[i]);
                feed_forward(n, n->input);

                /* print target vector (if available) */
                if (item->targets[i]) {
                        cprintf("T: ");
                        pprint == true
                                ? pprint_vector(item->targets[i], scheme)
                                : print_vector(item->targets[i]);
                }

                /* print output vector */
                cprintf("O: ");
                pprint == true
                        ? pprint_vector(n->output->vector, scheme)
                        : print_vector(n->output->vector);

                /* only compute and print error for last event */
                if (!(i == item->num_events - 1) || !item->targets[i])
                        continue;

                struct group *g = n->output;
                struct vector *tv = item->targets[i];
                double tr = n->target_radius;
                double zr = n->zero_error_radius; 

                /* compute and print error */
                n->status->error += n->output->err_fun->fun(g, tv, tr, zr);
                cprintf("\n");
                cprintf("Error:\t%lf\n", n->status->error);
        }

        cprintf("\n");
}

void test_rnn_network_with_item(struct network *n, struct item *item,
        bool pprint, uint32_t scheme)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        un->sp = 0;
        n->status->error = 0.0;
        
        cprintf("\n");
        cprintf("Name:   \"%s\"\n", item->name);
        cprintf("Meta:   \"%s\"\n", item->meta);
        cprintf("Events: %d\n", item->num_events);
        cprintf("\n");
        cprintf("(E: Event; I: Input; T: Target; O: Output)\n");

        reset_recurrent_groups(un->stack[un->sp]);
        for (uint32_t i = 0; i < item->num_events; i++) {
                /* print event number, and input vector */
                cprintf("\n");
                cprintf("E:  %d\n", i + 1);
                cprintf("I:  ");
                pprint == true
                        ? pprint_vector(item->inputs[i], scheme)
                        : print_vector(item->inputs[i]);

                /* feed activation vector */
                copy_vector(un->stack[un->sp]->input->vector, item->inputs[i]);
                feed_forward(un->stack[un->sp], un->stack[un->sp]->input);

                /* print target vector (if available) */
                if (item->targets[i]) {
                        cprintf("T: ");
                        pprint == true
                                ? pprint_vector(item->targets[i], scheme)
                                : print_vector(item->targets[i]);
                }

                /* print output vector */
                cprintf("O: ");
                pprint == true
                        ? pprint_vector(un->stack[un->sp]->output->vector, scheme)
                        : print_vector(un->stack[un->sp]->output->vector);

                /*
                 * Only compute and print error if there is a target for the
                 * current event.
                 */
                if (!item->targets[i])
                        goto shift_stack;

                struct group *g = un->stack[un->sp]->output;
                struct vector *tv = item->targets[i];
                double tr = n->target_radius;
                double zr = n->zero_error_radius;

                /* compute and print error */
                n->status->error += n->output->err_fun->fun(g, tv, tr, zr);
                cprintf("\n");
                cprintf("Error:\t%lf\n", n->status->error);

shift_stack:
                un->sp == un->stack_size - 1
                        ? rnn_shift_stack(un)
                        : un->sp++;
        }
        
        cprintf("\n");
}

void print_testing_summary(struct network *n, uint32_t tr)
{
        cprintf("\n");
        cprintf("Number of items: \t\t %d\n",
                        n->asp->items->num_elements);
        cprintf("Total error: \t\t\t %lf\n",
                        n->status->error);
        cprintf("Error per example: \t\t %lf\n",
                        n->status->error / n->asp->items->num_elements);
        cprintf("# Items reached threshold: \t %d (%.2lf%%)\n",
                        tr, ((double)tr / n->asp->items->num_elements) * 100.0);
        cprintf("\n");
}

void testing_signal_handler(int32_t signal)
{
        cprintf("Testing interrupted. Abort [y/n]");
        int32_t c = getc(stdin);
        getc(stdin); /* get newline */
        if (c == 'y' || c == 'Y')
                keep_running = false;
}
