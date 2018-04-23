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
#include "error.h"
#include "main.h"
#include "pprint.h"
#include "test.h"

static bool keep_running = true;

                /**********************
                 **** test network ****
                 **********************/

void test_network(struct network *n, bool verbose)
{
        struct sigaction sa;
        sa.sa_handler = testing_signal_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;
        sigaction(SIGINT, &sa, NULL);

        keep_running = true;

        switch (n->type) {
        case ntype_ffn: /* fall through */
        case ntype_srn:
                test_ffn_network(n, verbose);
                break;
        case ntype_rnn:
                test_rnn_network(n, verbose);
                break;
        }

        sa.sa_handler = SIG_DFL;
        sigaction(SIGINT, &sa, NULL);
}

void test_ffn_network(struct network *n, bool verbose)
{
        n->status->error = 0.0;
        uint32_t threshold_reached = 0;

        /* test network on all items in the current set */
        if (verbose) cprintf("\n");
        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                if (!keep_running) return;
                struct item *item = n->asp->items->elements[i];

                if (n->type == ntype_srn)
                        reset_context_groups(n);
                for (uint32_t j = 0; j < item->num_events; j++) {
                        if (j > 0 && n->type == ntype_srn)
                                shift_context_groups(n);
                        copy_vector(n->input->vector, item->inputs[j]);
                        feed_forward(n, n->input);

                        /* only compute network error for last event */
                        if (!(item->targets[j] && j == item->num_events - 1))
                                continue;

                        struct group *g   = n->output;
                        struct vector *tv = item->targets[j];
                        double tr         = n->target_radius;
                        double zr         = n->zero_error_radius;

                        double error = n->output->err_fun->fun(g, tv, tr, zr);
                        n->status->error += error;
                        if (error <= n->error_threshold)
                                threshold_reached++;

                        if (!verbose)
                                continue;
                        error <= n->error_threshold
                                ? pprintf("%d: \x1b[32m%s: %f\x1b[0m\n",
                                        i + 1, item->name, error)
                                : pprintf("%d: \x1b[31m%s: %f\x1b[0m\n",
                                        i + 1, item->name, error);
                }
        }

        print_testing_summary(n, threshold_reached);
}

void test_rnn_network(struct network *n, bool verbose)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        n->status->error = 0.0;
        uint32_t threshold_reached = 0;

        /* test network on all items in the current set */
        if (verbose) cprintf("\n");
        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                if (!keep_running) return;
                struct item *item = n->asp->items->elements[i];

                reset_recurrent_groups(un->stack[un->sp]);
                for (uint32_t j = 0; j < item->num_events; j++) {
                        copy_vector(
                                un->stack[un->sp]->input->vector,
                                item->inputs[j]);
                        feed_forward(
                                un->stack[un->sp],
                                un->stack[un->sp]->input);

                        /* only compute error if event has a target */
                        if (!item->targets[j]) goto next_tick;

                        struct group *g   = un->stack[un->sp]->output;
                        struct vector *tv = item->targets[j];
                        double tr         = n->target_radius;
                        double zr         = n->zero_error_radius;

                        double error = n->output->err_fun->fun(g, tv, tr, zr);
                        n->status->error += error;
                        if (error <= n->error_threshold)
                                threshold_reached++;

                        if (!verbose)
                                continue;
                        error <= n->error_threshold
                                ? pprintf("%d: \x1b[32m%s: %f\x1b[0m\n",
                                        i + 1, item->name, error)
                                : pprintf("%d: \x1b[31m%s: %f\x1b[0m\n",
                                        i + 1, item->name, error);

next_tick:
                        shift_pointer_or_stack(n);
                }
        }

        print_testing_summary(n, threshold_reached);
}

                /********************************
                 **** test network with item ****
                 ********************************/

void test_network_with_item(struct network *n, struct item *item,
        bool pprint, enum color_scheme scheme)
{
        switch (n->type) {
        case ntype_ffn: /* fall through */
        case ntype_srn:
                test_ffn_network_with_item(n, item, pprint, scheme);
                break;
        case ntype_rnn:
                test_rnn_network_with_item(n, item, pprint, scheme);
                break;
        }
}

void test_ffn_network_with_item(struct network *n, struct item *item,
        bool pprint, enum color_scheme scheme)
{
        n->status->error = 0.0;

        cprintf("\n");
        cprintf("Name:   \"%s\"\n", item->name);
        cprintf("Meta:   \"%s\"\n", item->meta);
        cprintf("Events: %d\n", item->num_events);
        cprintf("\n");
        cprintf("(E: Event; I: Input; T: Target; O: Output)\n");

        if (n->type == ntype_srn)
                reset_context_groups(n);
        for (uint32_t i = 0; i < item->num_events; i++) {
                if (i > 0 && n->type == ntype_srn)
                shift_context_groups(n);
                        copy_vector(n->input->vector, item->inputs[i]);
                        feed_forward(n, n->input);

                cprintf("\n");
                cprintf("E: %d\n", i + 1);
                cprintf("I: ");
                pprint ? pprint_vector(item->inputs[i], scheme)
                       : print_vector(item->inputs[i]);
                if (item->targets[i]) {
                        cprintf("T: ");
                        pprint ? pprint_vector(item->targets[i], scheme)
                               : print_vector(item->targets[i]);
                }
                cprintf("O: ");
                pprint ? pprint_vector(n->output->vector, scheme)
                       : print_vector(n->output->vector);

                /* only compute and print error for last event */
                if (!(i == item->num_events - 1) || !item->targets[i])
                        continue;

                struct group *g   = n->output;
                struct vector *tv = item->targets[i];
                double tr         = n->target_radius;
                double zr         = n->zero_error_radius; 

                n->status->error += n->output->err_fun->fun(g, tv, tr, zr);
                cprintf("\nError:\t%lf\n", n->status->error);
        }

        cprintf("\n");
}

void test_rnn_network_with_item(struct network *n, struct item *item,
        bool pprint, enum color_scheme scheme)
{
        struct rnn_unfolded_network *un = n->unfolded_net;
        n->status->error = 0.0;
        
        cprintf("\n");
        cprintf("Name:   \"%s\"\n", item->name);
        cprintf("Meta:   \"%s\"\n", item->meta);
        cprintf("Events: %d\n", item->num_events);
        cprintf("\n");
        cprintf("(E: Event; I: Input; T: Target; O: Output)\n");

        reset_stack_pointer(n);
        reset_recurrent_groups(un->stack[un->sp]);
        for (uint32_t i = 0; i < item->num_events; i++) {
                copy_vector(
                        un->stack[un->sp]->input->vector,
                        item->inputs[i]);
                feed_forward(
                        un->stack[un->sp],
                        un->stack[un->sp]->input);

                cprintf("\n");
                cprintf("E: %d\n", i + 1);
                cprintf("I: ");
                pprint ? pprint_vector(item->inputs[i], scheme)
                       : print_vector(item->inputs[i]);
                if (item->targets[i]) {
                        cprintf("T: ");
                        pprint ? pprint_vector(item->targets[i], scheme)
                               : print_vector(item->targets[i]);
                }
                cprintf("O: ");
                pprint ? pprint_vector(un->stack[un->sp]->output->vector,
                                scheme)
                       : print_vector(un->stack[un->sp]->output->vector);

                /* only compute error if event has a target */
                if (!item->targets[i]) goto next_tick;

                struct group *g   = un->stack[un->sp]->output;
                struct vector *tv = item->targets[i];
                double tr         = n->target_radius;
                double zr         = n->zero_error_radius;

                n->status->error += n->output->err_fun->fun(g, tv, tr, zr);
                cprintf("\nError:\t%lf\n", n->status->error);

next_tick:
                shift_pointer_or_stack(n);
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
        if (n->output->err_fun->fun == err_fun_sum_of_squares)
                cprintf("Root Mean Square (RMS) error: \t %lf\n",
                        sqrt((2.0 * n->status->error) / n->asp->items->num_elements));
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
