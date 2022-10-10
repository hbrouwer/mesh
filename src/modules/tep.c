/*
 * Copyright 2012-2022 Harm Brouwer <me@hbrouwer.eu>
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

#include "tep.h"

#include "../engine.h"
#include "../main.h"
#include "../math.h"
#include "../vector.h"

static bool keep_running = true;

                /*****************************************
                 **** temporally extended propagation ****
                 *****************************************/

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
This implements temporally extended activation propagation for SRNs. After
processing time-step t, the activation pattern at the hidden layer will be
the optimal context for time-step t+1. At a given time-step t, we therefore
model how we dynamically move over time from the [current state] (context at
t-1) to the [next state] (context at t), the optimal context for processing
t, while processing the input at t. At t=0 this is bootstrapped by setting
the current state to be the unit vector v(1) / |v(1)|.

Consider the states after processing time-step t-1:

                                       +---------+      | [output]
                                       | output  | <--- |
                                       +---------+      | (output at t-1)
                                            |
                           +-------------+  |
                           |             |  |
                           |           +---------+      | [next state]
                           |           | hidden  | <--- | 
                           |           +---------+      | (context at t)
                           |             |  |
                           |     +-------+  |
                           |     |          |
  [current state] |      +---------+   +---------+      | [input]
                  | ---> | context |   |  input  | <--- |
 (context at t-1) |      +---------+   +---------+      | (input at t-1)

Given the next input at time-step t, we model how the [current state]
(context at t-1) dynamically moves into its [next state] (context at t), and
how this affects the output while processing the input at t.

The dynamic movement from the [current state] (context at t-1) to the [next
state] (context at t) is modeled as a time-invariant 4th order "classic"
Runge-Kutta approximation. The [current state] will iteratively move to the
[next state] in discrete micro time-steps with step-size "h". Each [current
state] is fed-forward through the network to compute the output vector. This
is repeated for "n" iterations, until the cosine distance between the
previous and current output vector is smaller than a "th" parameter. The
total processing time, the number of micro time-steps, is then the number of
iterations "n" times step-size "h".
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

double tep_iterate(struct network *n, struct group *eg, double h, double th,
        struct vector *cs, struct vector *ns,
        /* - - for recording - - */
        struct group *rg, uint32_t item_num, struct item *item,
        uint32_t event_num, FILE *fd)
{
        struct vector *po = create_vector(n->output->vector->size);

        /*
         * Move from the [current state] to the [next state], using the 4th
         * order "classic" Runge-Kutta method solving:
         *
         *      d_y / d_t = f(t,y)
         *
         * We do, however, assume a time-invariant system in that f is
         * indepedent of t, and we define f as:
         *
         *      f(y) = [next state] - [current state]
         *
         * We increment t at the end of each iteration to give us a notion
         * of time.
         *
         * At each iteration the [current state] is injected, and activation
         * is propagated forward. When the cosine distance between the
         * previous and current output vector is smaller than the "th"
         * parameter, the actual [next state] is injected to assure output
         * equivalence to non-temporally extended propagation. This takes
         * one additional micro time-step.
         */
        
        double mt = 0.0;
        bool th_reached = false;
        while (!th_reached) {
                /* Runge-Kutta iterations */
                if (1.0 - cosine(po, n->output->vector) >= th) {
                        copy_vector(n->output->vector, po);
                        
                        for (uint32_t j = 0; j < cs->size; j++) {
                                double cu = cs->elements[j];
                                double nu = ns->elements[j];
                                
                                /* k1 = f(y_t) = ns - cs */
                                double k1 = nu - cu;
                                /* k2 = f(y_t + h * (k1 / 2)) */
                                double k2 = nu - (cu + h * (k1 / 2.0));
                                /* k3 = f(y_t + h * (k2 / 2)) */
                                double k3 = nu - (cu + h * (k2 / 2.0));
                                /* k4 = f(y_t + h * k3) */
                                double k4 = nu - (cu + h * k3);
                                
                                /* 
                                 * dy = (1/6) * (k1 + (2 * k2) + (2 * k3) + k4) * h
                                 *
                                 * y_t+1 = y_t + dy
                                 */
                                double dy = 1.0 / 6.0;
                                dy *= k1 + 2.0 * k2 + 2.0 * k3 + k4;
                                dy *= h;
                                cs->elements[j] += dy;
                        }
                        mt += h;
                        
                        /* inject [current state] and update network */
                        copy_vector(cs, eg->vector);
                        next_tick(n);
                        forward_sweep(n);
                /* final time-step */
                } else {
                        th_reached = true;
                        mt += h;
                        
                        /*
                         * Inject actual [next state] and update network to
                         * assure output equivalence to non-temporally
                         * extended propagation.
                         */
                        copy_vector(ns, eg->vector);
                        next_tick(n);
                        forward_sweep(n);
                }
                
                /* record units (if required) */
                if (rg != NULL) {
                        fprintf(fd, "%d,\"%s\",\"%s\",%d,%s,%f",
                                item_num, item->name, item->meta, event_num, rg->name, mt);
                        for (uint32_t u = 0; u < rg->vector->size; u++)
                                fprintf(fd, ",%f", rg->vector->elements[u]);
                        fprintf(fd, "\n");
                }
        }

        free_vector(po);

        return mt;
}

void tep_test_network_with_item(struct network *n, struct group *eg, double h,
        double th, struct item *item, bool pprint,
        enum color_scheme scheme)
{
        n->status->error = 0.0;

        cprintf("\n");
        cprintf("Name:   \"%s\"\n", item->name);
        cprintf("Meta:   \"%s\"\n", item->meta);
        cprintf("Events: %d\n", item->num_events);
        cprintf("\n");
        cprintf("(E: Event; I: Input; T: Target; O: Output)\n");
        
        /* [current state] */
        struct vector *cs = create_vector(eg->vector->size);
        /* [next state] */
        struct vector *ns = create_vector(eg->vector->size);

        reset_ticks(n);
        for (uint32_t i = 0; i < item->num_events; i++) {
                /*
                 * Current [next state] is the new [current state]. At t=0,
                 * the [current state] is the unit vector v(1) / |v(1)|.
                 *
                 * As the hidden layer activation pattern is shifted into
                 * the context layer, the new [next state] is that of the
                 * context group.
                 */
                if (i == 0) {
                        fill_vector_with_value(ns, 1.0);
                        fill_vector_with_value(ns, 1.0 / euclidean_norm(ns));
                }
                copy_vector(ns, cs);
                if (i > 0)
                        next_tick(n);
                clamp_input_vector(n, item->inputs[i]);
                forward_sweep(n);
                struct group *cg = eg->ctx_groups->elements[0];
                copy_vector(cg->vector, ns);

                /* move from the [current state] to the [next state] */
                double mt = tep_iterate(n, eg, h, th, cs, ns, NULL, 0, NULL, 0, NULL);
                
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
                cprintf("MicroTicks: %f\n", mt);
                
                if (!(item->targets[i] && i == item->num_events - 1))
                        continue;
                n->status->error += output_error(n, item->targets[i]);
                cprintf("\nError:\t%lf\n", n->status->error);
                cprintf("\n");
        }

        free_vector(ns);
        free_vector(cs);
}

void tep_record_units(struct network *n, struct group *eg, double h,
        double th, struct group *rg, char *filename)
{
        struct sigaction sa;
        sa.sa_handler = tep_signal_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;
        sigaction(SIGINT, &sa, NULL);
        
        FILE *fd;
        if (!(fd = fopen(filename, "w")))
                goto error_out;

        fprintf(fd, "\"ItemId\",\"ItemName\",\"ItemMeta\",\"EventNum\",\"Group\",\"MicroTick\"");
        for (uint32_t u = 0; u < rg->vector->size; u++)
                fprintf(fd, ",\"Unit%d\"", u + 1);
        fprintf(fd, "\n");
        
        /* [current state] */
        struct vector *cs = create_vector(eg->vector->size);
        /* [next state] */
        struct vector *ns = create_vector(eg->vector->size);
        
        cprintf("\n");
        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                if (!keep_running) {
                        keep_running = true;
                        goto out;
                }
                struct item *item = n->asp->items->elements[i];
                zero_out_vector(cs);
                zero_out_vector(ns);
                reset_ticks(n);
                for (uint32_t j = 0; j < item->num_events; j++) {
                        /*
                         * Current [next state] is the new [current state].
                         * At t=0, the [current state] is the unit vector
                         * v(1) / |v(1)|.
                         *
                         * As the hidden layer activation pattern is shifted
                         * into the context layer, the new [next state] is
                         * that of the context group.
                         */
                        if (j == 0) {
                                fill_vector_with_value(ns, 1.0);
                                fill_vector_with_value(ns, 1.0 / euclidean_norm(ns));
                        }
                        copy_vector(ns, cs);
                        if (j > 0)
                                next_tick(n);
                        clamp_input_vector(n, item->inputs[j]);
                        forward_sweep(n);
                        struct group *cg = eg->ctx_groups->elements[0];
                        copy_vector(cg->vector, ns);
                
                        /* move from the [current state] to the [next state] */
                        tep_iterate(n, eg, h, th, cs, ns, rg, i + 1, item, j + 1, fd);
                }
                pprintf("%d: %s\n", i + 1, item->name);
        }
        cprintf("\n");
        
out:
        free_vector(ns);
        free_vector(cs);

        fclose(fd);
        
        sa.sa_handler = SIG_DFL;
        sigaction(SIGINT, &sa, NULL);

        return;

error_out:
        perror("[tep_record_units()]");
        return;
}

void tep_write_micro_ticks(struct network *n, struct group *eg, double h,
        double th, char *filename)
{
        struct sigaction sa;
        sa.sa_handler = tep_signal_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;
        sigaction(SIGINT, &sa, NULL);
        
        FILE *fd;
        if (!(fd = fopen(filename, "w")))
                goto error_out;
        
        cprintf("\n");
        fprintf(fd, "\"ItemId\",\"ItemName\",\"ItemMeta\",\"EventNum\",\"MicroTicks\"\n");
        for (uint32_t i = 0; i < n->asp->items->num_elements; i++) {
                if (!keep_running) {
                        keep_running = true;
                        goto out;
                }
                struct item *item = n->asp->items->elements[i];
                struct vector *muticks = tep_micro_ticks_for_item(n, eg, h, th, item);
                for (uint32_t j = 0; j < item->num_events; j++) 
                        fprintf(fd,"%d,\"%s\",\"%s\",%d,%f\n",
                                i + 1, item->name, item->meta, j + 1, muticks->elements[j]);
                pprintf("%d: %s\n", i + 1, item->name);
                free_vector(muticks);
        }
        cprintf("\n");

out:
        fclose(fd);
        
        sa.sa_handler = SIG_DFL;
        sigaction(SIGINT, &sa, NULL);

        return;

error_out:
        perror("[tep_write_micro_ticks()]");
        return;
}

struct vector *tep_micro_ticks_for_item(struct network *n, struct group *eg,
        double h, double th, struct item *item)
{       
        struct vector *muticks = create_vector(item->num_events);

        /* [current state] */
        struct vector *cs = create_vector(eg->vector->size);
        /* [next state] */
        struct vector *ns = create_vector(eg->vector->size);
       
        reset_ticks(n);
        for (uint32_t i = 0; i < item->num_events; i++) {
                /*
                 * Current [next state] is the new [current state]. At t=0,
                 * the [current state] is the unit vector v(1) / |v(1)|.
                 *
                 * As the hidden layer activation pattern is shifted into
                 * the context layer, the new [next state] is that of the
                 * context group.
                 */
                if (i == 0) {
                        fill_vector_with_value(ns, 1.0);
                        fill_vector_with_value(ns, 1.0 / euclidean_norm(ns));
                }
                copy_vector(ns, cs);
                if (i > 0)
                        next_tick(n);
                clamp_input_vector(n, item->inputs[i]);
                forward_sweep(n);
                struct group *cg = eg->ctx_groups->elements[0];
                copy_vector(cg->vector, ns);

                /* move from the [current state] to the [next state] */
                muticks->elements[i] = tep_iterate(n, eg, h, th, cs, ns, NULL, 0, NULL, 0, NULL);
        }

        free_vector(ns);
        free_vector(cs);

        return(muticks);
}

void tep_signal_handler(int32_t signal)
{
        cprintf("(interrupted): Abort [y/n]? ");
        int32_t c = getc(stdin);
        getc(stdin); /* get newline */
        if (c == 'y' || c == 'Y')
                keep_running = false;
}
