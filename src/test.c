/*
 * Copyright 2012-2019 Harm Brouwer <me@hbrouwer.eu>
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

#include "engine.h"
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

        n->status->error = 0.0;
        uint32_t tr      = 0;
        if (verbose)
                cprintf("\n");
        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                if (!keep_running)
                        return;
                struct item *item = n->asp->items->elements[i];
                reset_ticks(n);
                for (uint32_t j = 0; j < item->num_events; j++) {
                        if (j > 0)
                                next_tick(n);
                        clamp_input_vector(n, item->inputs[j]);
                        forward_sweep(n);
                        if (!(item->targets[j] && j == item->num_events - 1))
                                continue;
                        double error = output_error(n, item->targets[j]);
                        n->status->error += error;
                        if (error <= n->pars->error_threshold)
                                tr++;
                        if (!verbose)
                                continue;
                        error <= n->pars->error_threshold
                                ? pprintf("%d: \x1b[32m%s: %f\x1b[0m\n",
                                        i + 1, item->name, error)
                                : pprintf("%d: \x1b[31m%s: %f\x1b[0m\n",
                                        i + 1, item->name, error);
                }
        }
        
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

        sa.sa_handler = SIG_DFL;
        sigaction(SIGINT, &sa, NULL);
}

                /********************************
                 **** test network with item ****
                 ********************************/

void test_network_with_item(struct network *n, struct item *item,
        bool pprint, enum color_scheme scheme)
{
        n->status->error = 0.0;

        cprintf("\n");
        cprintf("Name:   \"%s\"\n", item->name);
        cprintf("Meta:   \"%s\"\n", item->meta);
        cprintf("Events: %d\n", item->num_events);
        cprintf("\n");
        cprintf("(E: Event; I: Input; T: Target; O: Output)\n");

        reset_ticks(n);
        for (uint32_t i = 0; i < item->num_events; i++) {
                if(i > 0)
                        next_tick(n);
                clamp_input_vector(n, item->inputs[i]);
                forward_sweep(n);
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
                struct vector *ov = output_vector(n);
                pprint ? pprint_vector(ov, scheme)
                       : print_vector(ov);

                if (!(item->targets[i] && i == item->num_events - 1))
                        continue;
                n->status->error += output_error(n, item->targets[i]);
                cprintf("\nError:\t%lf\n", n->status->error);
                cprintf("\n");
        }
}

void testing_signal_handler(int32_t signal)
{
        cprintf("Testing interrupted. Abort [y/n]");
        int32_t c = getc(stdin);
        getc(stdin); /* get newline */
        if (c == 'y' || c == 'Y')
                keep_running = false;
}
